from enum import Enum
from typing import Any

from llama_stack.distribution.datatypes import Api

from llama_stack.providers.inline.post_training.torchtune.post_training import TorchtunePostTrainingImpl
from llama_stack.providers.inline.post_training.torchtune.config import TorchtunePostTrainingConfig
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    Checkpoint,
    JobStatus,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)

from llama_stack.schema_utils import webmethod

from .scheduler import JobArtifact, Scheduler
from .scheduler import JobStatus as SchedulerJobStatus
from .pipeline import pipeline, PipelineMode


class TrainingArtifactType(Enum):
    CHECKPOINT = "checkpoint"
    RESOURCES_STATS = "resources_stats"


_JOB_TYPE_SUPERVISED_FINE_TUNE = "supervised-fine-tune"


class TorchtuneKFPPostTrainingImpl(TorchtunePostTrainingImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO: make it configurable
        #self._mode = PipelineMode.LOCAL
        self._mode = PipelineMode.REMOTE
        self._scheduler = Scheduler(backend=f"kfp-{self._mode.value}", to_artifacts=self._to_artifacts)

    async def shutdown(self) -> None:
        await self._scheduler.shutdown()

    @classmethod
    def _to_artifacts(cls, in_artifact) -> list[JobArtifact]:
        return [
            cls._checkpoint_to_artifact(Checkpoint.model_validate(checkpoint))
            for checkpoint in in_artifact.metadata['checkpoints']
        ] + [
            cls._resources_stats_to_artifact(in_artifact.metadata['resources_allocated'])
        ]

    @staticmethod
    def _checkpoint_to_artifact(checkpoint: Checkpoint) -> JobArtifact:
        return JobArtifact(
            type=TrainingArtifactType.CHECKPOINT.value,
            name=checkpoint.identifier,
            uri=checkpoint.path,
            metadata=dict(checkpoint),
        )

    @staticmethod
    def _resources_stats_to_artifact(resources_stats: dict[str, Any]) -> JobArtifact:
        return JobArtifact(
            type=TrainingArtifactType.RESOURCES_STATS.value,
            name=TrainingArtifactType.RESOURCES_STATS.value,
            metadata=resources_stats,
        )

    async def _get_all_data(self, dataset_id: str) -> list[dict[str, Any]]:
        all_rows = await self.datasetio_api.iterrows(dataset_id=dataset_id, limit=-1)
        return all_rows.data

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None,
        algorithm_config: AlgorithmConfig | None,
    ) -> PostTrainingJob:
        if not isinstance(algorithm_config, LoraFinetuningConfig):
            raise NotImplementedError()

        data = await self._get_all_data(training_config.data_config.dataset_id)
        p = pipeline(
            self._mode,
            self.config,
            data,
            job_uuid,
            training_config,
            hyperparam_search_config,
            logger_config,
            model,
            checkpoint_dir or "null",
            algorithm_config,
        )

        job_uuid = self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, p)
        return PostTrainingJob(job_uuid=job_uuid)

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(
            data=[PostTrainingJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )

    @staticmethod
    def _get_artifacts_metadata_by_type(job, artifact_type):
        return [artifact.metadata for artifact in job.artifacts if artifact.type == artifact_type]

    @classmethod
    def _get_checkpoints(cls, job):
        return cls._get_artifacts_metadata_by_type(job, TrainingArtifactType.CHECKPOINT.value)

    @classmethod
    def _get_resources_allocated(cls, job):
        data = cls._get_artifacts_metadata_by_type(job, TrainingArtifactType.RESOURCES_STATS.value)
        return data[0] if data else None

    @webmethod(route="/post-training/job/status")
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        job = self._scheduler.get_job(job_uuid)

        match job.status:
            # TODO: Add support for other statuses to API
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                raise NotImplementedError()

        return PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=status,
            scheduled_at=job.scheduled_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            checkpoints=self._get_checkpoints(job),
            resources_allocated=self._get_resources_allocated(job),
        )

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None:
        self._scheduler.cancel(job_uuid)

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        job = self._scheduler.get_job(job_uuid)
        return PostTrainingJobArtifactsResponse(job_uuid=job_uuid, checkpoints=self._get_checkpoints(job))


async def get_adapter_impl(config: TorchtunePostTrainingConfig, _deps):
    return TorchtuneKFPPostTrainingImpl(config, _deps[Api.datasetio], _deps[Api.datasets])
