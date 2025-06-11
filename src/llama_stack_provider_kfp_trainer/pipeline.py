# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
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


_ACCELERATOR_TYPE = "nvidia.com/gpu"


def _get_provider_pip_dependencies(api_type: Api, provider_name: str) -> list[str]:
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
        if name == provider_name:
            deps += spec.pip_packages

    # drop unnecessary dependencies
    # TODO: make it more generic / separate scheduler deps?
    deps.remove("kubernetes")

    return deps


def lls_component(
    config: TorchtuneKFPTrainerConfig,
    api_type: Api,
    provider_name: str,
    use_gpu: bool = False,
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            base_image = config.base_image
            component_obj = dsl.component(
                base_image=base_image,
                func=func,
                packages_to_install=_get_provider_pip_dependencies(
                    api_type, provider_name
                ),
            )(*args, **kwargs).set_memory_limit("16Gi")
            if use_gpu:
                component_obj = component_obj.set_accelerator_limit(
                    1
                ).set_accelerator_type(_ACCELERATOR_TYPE)
            return component_obj

        return wrapper

    return decorator


def get_component(config: TorchtuneKFPTrainerConfig, use_gpu: bool):
    return lls_component(config, Api.post_training, "kfp-torchtune", use_gpu=use_gpu)(
        component_impl
    )


def component_impl(
    config: dict,
    data_artifact: Input[Artifact],
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

    # Extract data from passed artifact
    import json

    with open(data_artifact.path) as f:
        data = json.load(f)
        print(f"Loaded {len(data)} rows of data from {data_artifact.path}")

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


def _dump_data(data: list[dict], job_uuid: str, root_dir: str) -> str:
    data_path = Path(root_dir) / f"{job_uuid}_data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    return str(data_path)


def _upload_data_to_s3(data: list[dict], job_uuid: str, s3_bucket: str) -> str:
    from .s3 import Client
    import uuid as _uuid

    local_data_path = _dump_data(data, job_uuid, "/tmp")
    dataset_object_name = f"{job_uuid}_data_{_uuid.uuid4().hex}.json"

    s3_client = Client(s3_bucket)
    s3_client.upload(Path(local_data_path), dataset_object_name)

    return f"s3://{s3_bucket}/{dataset_object_name}"


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
    # TODO: this sucks, but seems that directory name doesn't match model name?
    if model.startswith("Llama-"):
        model = model.replace("Llama-", "Llama")

    if config.mode == PipelineMode.LOCAL:
        artifact_prefix = str(Path(os.environ["HOME"]) / ".llama" / "checkpoints")
        data_uri = _dump_data(data, job_uuid, artifact_prefix)
        use_gpu = False
    else:
        data_uri = _upload_data_to_s3(data, job_uuid, config.s3_bucket)
        artifact_prefix = f"s3://{config.s3_bucket}"
        use_gpu = getattr(config, "use_gpu", False)

    fname = f"{model}.tar.gz"
    artifact_uri = f"{artifact_prefix}/{fname}"

    component = get_component(config, use_gpu)

    @dsl.pipeline(name=job_uuid)
    def p(
        artifact_uri: str = artifact_uri,
        config: dict = _serialize(config),
        data_uri: str = data_uri,
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

        data = dsl.importer(
            artifact_uri=data_uri,
            artifact_class=dsl.Dataset,
        )

        return component(
            config=config,
            data_artifact=data.output,
            job_uuid=job_uuid,
            training_config=training_config,
            hyperparam_search_config=hyperparam_search_config,
            logger_config=logger_config,
            model=model,
            algorithm_config=algorithm_config,
            model_artifact=a.output,
        ).output

    return p
