import enum

from llama_stack.providers.inline.post_training.torchtune import (
    TorchtunePostTrainingConfig,
)


class PipelineMode(enum.Enum):
    LOCAL = "local"
    REMOTE = "remote"


class TorchtuneKFPTrainerConfig(TorchtunePostTrainingConfig):
    """
    Configuration for KFP Torchtune Trainer.
    """

    mode: PipelineMode = PipelineMode.LOCAL
    s3_bucket: str
    use_gpu: bool = False  # If True, enable GPU for remote training. Default: CPU only.
