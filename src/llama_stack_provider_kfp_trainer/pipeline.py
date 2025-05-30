# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
import os

from kfp import dsl
from kfp.dsl import Artifact, Input, Output
from pydantic import BaseModel

from llama_stack.apis.post_training import (
    LoraFinetuningConfig,
    TrainingConfig,
)
from llama_stack.apis.datatypes import Api
from llama_stack.distribution.distribution import get_provider_registry


from .config import PipelineMode, TorchtuneKFPTrainerConfig
from .provider import get_provider_spec


def _get_provider_pip_dependencies(
    api_type: Api, provider_name: str | None = None
) -> list[str]:
    deps = [
        # TODO: how to achieve identical llama-stack code on both sides?
        "git+https://github.com/meta-llama/llama-stack.git@main#egg=llama-stack",
    ]  # always install the base package
    # TODO: get_provider_registry should return external providers too?
    provider_registry = get_provider_registry() | {
        Api.post_training: {
            "kfp-torchtune": get_provider_spec().adapter,
        },
    }
    for name, spec in provider_registry[api_type].items():
        if provider_name is None or name == provider_name:
            deps += spec.pip_packages

    # drop unnecessary dependencies
    # TODO: make it more generic / separate scheduler deps?
    deps.remove("kfp")
    deps.remove("kubernetes")

    return deps


# TODO: we should probably have a container image with all dependencies pre-built
_BASE_IMAGE = "quay.io/fedora/python-311:311"


def lls_component(api_type: Api, provider_name: str | None = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return dsl.component(
                base_image=_BASE_IMAGE,
                func=func,
                packages_to_install=_get_provider_pip_dependencies(
                    api_type, provider_name
                ),
            )(*args, **kwargs)

        return wrapper

    return decorator


@lls_component(Api.post_training, "kfp-torchtune")
def component(
    config: dict,
    data: list,  # should be an Input?
    job_uuid: str,
    training_config: dict,
    hyperparam_search_config: dict,
    logger_config: dict,
    model: str,
    algorithm_config: dict,
    model_artifact: Input[Artifact],
    output: Output[Artifact],
):
    from llama_stack.apis.post_training import (
        LoraFinetuningConfig,
        TrainingConfig,
    )
    from llama_stack.providers.inline.post_training.torchtune.config import (
        TorchtunePostTrainingConfig,
    )
    from llama_stack.providers.inline.post_training.torchtune.recipes.lora_finetuning_single_device import (
        LoraFinetuningSingleDevice,
    )

    # Note: Monkey patch LoraFinetuningSingleDevice to avoid callbacks to LLS
    # API for dataset - this is to avoid copying provider code from
    # llama-stack; not needed in a realistic provider.
    from functools import partial
    from torchtune.data import padded_collate_sft
    from llama_stack.providers.inline.post_training.torchtune.common import utils
    from llama_stack.providers.inline.post_training.torchtune.datasets.sft import (
        SFTDataset,
    )
    from torch.utils.data import DataLoader, DistributedSampler

    class MyLoraFinetuningSingleDevice(LoraFinetuningSingleDevice):
        def __init__(self, data, *args, **kwargs):
            super().__init__(*args, datasetio_api=None, datasets_api=None, **kwargs)
            self.data = data

        # override to use data passed by the pipeline
        async def _setup_data(
            self,
            dataset_id: str,
            tokenizer,
            shuffle: bool,
            batch_size: int,
        ):
            data_transform = await utils.get_data_transform(self._data_format)
            ds = SFTDataset(
                self.data,
                message_transform=data_transform(train_on_input=self._train_on_input),
                model_transform=tokenizer,
                dataset_type=self._data_format.value,
            )

            sampler = DistributedSampler(
                ds,
                num_replicas=1,
                rank=0,
                shuffle=shuffle,
                seed=0,
            )
            dataloader = DataLoader(
                dataset=ds,
                sampler=sampler,
                batch_size=batch_size,
                # dropping last avoids shape issues with compile + flex attention
                drop_last=True,
                collate_fn=(
                    partial(
                        padded_collate_sft,
                        padding_idx=self._tokenizer.pad_id,
                        ignore_idx=self._loss_fn.ignore_index,
                    )
                ),
            )

            return sampler, dataloader

    #### End of monkey patching

    # Extract checkpoint from passed artifact
    import os
    import tarfile

    model_dir = os.path.dirname(model_artifact.path)

    dest_dir = os.path.join(model_dir, "model_artifact")
    with tarfile.open(model_artifact.path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    recipe = MyLoraFinetuningSingleDevice(
        data,
        TorchtunePostTrainingConfig(**config),
        job_uuid,
        TrainingConfig(**training_config),
        hyperparam_search_config,
        logger_config,
        model,
        dest_dir,
        LoraFinetuningConfig(**algorithm_config),
    )

    import asyncio

    asyncio.run(recipe.setup())
    resources_allocated, checkpoints = asyncio.run(recipe.train())

    # TODO: how does one reuse code with kfp?
    def _serialize(obj) -> dict:
        return obj.model_dump(exclude_none=True, mode="json")

    output.metadata = {
        "resources_allocated": resources_allocated,
        "checkpoints": [],
    }

    # Copy checkpoint files to pipeline artifacts
    import shutil

    for checkpoint in checkpoints:
        chk = checkpoint.model_copy()
        chk.path = f"{output.path}/{checkpoint.identifier}"
        output.metadata["checkpoints"].append(_serialize(chk))
        # TODO: handle any errors
        shutil.copytree(checkpoint.path, chk.path)


# TODO: should serialize use strings to pass models between components?
def _serialize(obj: BaseModel) -> dict:
    return obj.model_dump(exclude_none=True, mode="json")


# TODO: it would be nice if we could pass pydantic models transparently between
# components (with serialization and deserialization offloaded to kfp
# machinery): https://github.com/kubeflow/pipelines/issues/10690
def pipeline(
    config: TorchtuneKFPTrainerConfig,
    data: list[dict],
    job_uuid: str,
    training_config: TrainingConfig,
    hyperparam_search_config: dict,
    logger_config: dict,
    model: str,
    algorithm_config: LoraFinetuningConfig,
):
    # TODO: pass it through artifact to avoid issues with size
    data = data[:10]

    # TODO: this sucks, but seems that directory name doesn't match model name?
    if model.startswith("Llama-"):
        model = model.replace("Llama-", "Llama")

    if config.mode == PipelineMode.LOCAL:
        artifact_prefix = str(Path(os.environ["HOME"]) / ".llama" / "checkpoints")
    else:
        # TODO: make bucket configurable
        artifact_prefix = "s3://rhods-dsp-dev"

    fname = f"{model}.tar.gz"
    artifact_uri = f"{artifact_prefix}/{fname}"

    @dsl.pipeline(name=job_uuid)
    def p(
        artifact_uri: str = artifact_uri,
        config: dict = _serialize(config),
        data: list = data,
        job_uuid: str = job_uuid,
        training_config: dict = _serialize(training_config),
        hyperparam_search_config: dict = hyperparam_search_config,
        logger_config: dict = logger_config,
        model: str = model,
        algorithm_config: dict = _serialize(algorithm_config),
    ) -> Artifact:
        a = dsl.importer(
            artifact_uri=artifact_uri,
            artifact_class=dsl.Model,
        )

        return component(
            config=config,
            data=data,
            job_uuid=job_uuid,
            training_config=training_config,
            hyperparam_search_config=hyperparam_search_config,
            logger_config=logger_config,
            model=model,
            algorithm_config=algorithm_config,
            model_artifact=a.output,
        ).output

    return p
