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
