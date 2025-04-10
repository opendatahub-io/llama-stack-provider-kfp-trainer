from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    AdapterSpec,
    remote_provider_spec,
)


def get_provider_spec() -> ProviderSpec:
    return remote_provider_spec(
        api=Api.post_training,
        adapter=AdapterSpec(
            adapter_type="kfp-torchtune",
            # TODO: why do we duplicate what's already in yaml?
            pip_packages=["torch", "torchtune==0.5.0", "torchao==0.8.0", "numpy", "kfp", "kubernetes"],
            config_class="config.TorchtuneKFPPostTrainingConfig",
            module="kfp_adapter",
        ),
    )
