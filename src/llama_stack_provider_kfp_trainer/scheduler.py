# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import abc
import asyncio
import functools
import threading
import warnings
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Tuple, TypeAlias

from pydantic import BaseModel

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="scheduler")


# TODO: revisit the list of possible statuses when defining a more coherent
# Jobs API for all API flows; e.g. do we need new vs scheduled?
class JobStatus(Enum):
    new = "new"
    scheduled = "scheduled"
    running = "running"
    failed = "failed"
    completed = "completed"


JobID: TypeAlias = str
JobType: TypeAlias = str


class JobArtifact(BaseModel):
    type: JobType
    name: str
    # TODO: uri should be a reference to /files API; revisit when /files is implemented
    uri: str | None = None
    metadata: Dict[str, Any]


JobHandler = Any  # TODO: make it explicitly a pipeline type


LogMessage: TypeAlias = Tuple[datetime, str]


_COMPLETED_STATUSES = {JobStatus.completed, JobStatus.failed}


class Job:
    def __init__(self, job_type: JobType, job_id: JobID, handler: JobHandler):
        super().__init__()
        self.id = job_id
        self._provider_id: str | None = None
        self._type = job_type
        self._handler = handler
        self._artifacts: list[JobArtifact] = []
        self._logs: list[LogMessage] = []
        self._state_transitions: list[Tuple[datetime, JobStatus]] = [(datetime.now(timezone.utc), JobStatus.new)]

    @property
    def handler(self) -> JobHandler:
        return self._handler

    @property
    def status(self) -> JobStatus:
        return self._state_transitions[-1][1]

    @status.setter
    def status(self, status: JobStatus):
        if status in _COMPLETED_STATUSES and self.status in _COMPLETED_STATUSES:
            raise ValueError(f"Job is already in a completed state ({self.status})")
        if self.status == status:
            return
        self._state_transitions.append((datetime.now(timezone.utc), status))

    @property
    def artifacts(self) -> list[JobArtifact]:
        return self._artifacts

    def register_artifact(self, artifact: JobArtifact) -> None:
        self._artifacts.append(artifact)

    def _find_state_transition_date(self, status: Iterable[JobStatus]) -> datetime | None:
        for date, s in reversed(self._state_transitions):
            if s in status:
                return date
        return None

    @property
    def scheduled_at(self) -> datetime | None:
        return self._find_state_transition_date([JobStatus.scheduled])

    @property
    def started_at(self) -> datetime | None:
        return self._find_state_transition_date([JobStatus.running])

    @property
    def completed_at(self) -> datetime | None:
        return self._find_state_transition_date(_COMPLETED_STATUSES)

    @property
    def logs(self) -> list[LogMessage]:
        return self._logs[:]

    def append_log(self, message: LogMessage) -> None:
        self._logs.append(message)

    # TODO: implement
    def cancel(self) -> None:
        raise NotImplementedError

    @property
    def provider_id(self) -> str | None:
        return self._provider_id

    @provider_id.setter
    def provider_id(self, provider_id: str) -> None:
        if self._provider_id is not None:
            raise ValueError(f"Job {self.id} already has a provider id ({self._provider_id})")
        self._provider_id = provider_id


class _SchedulerBackend(abc.ABC):
    @abc.abstractmethod
    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        raise NotImplementedError


class _NaiveSchedulerBackend(_SchedulerBackend):
    def __init__(self, timeout: int = 5):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        # There may be performance implications of using threads due to Python
        # GIL; may need to measure if it's a real problem though
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

        # When stopping the loop, give tasks a chance to finish
        # TODO: should we explicitly inform jobs of pending stoppage?
        for task in asyncio.all_tasks(self._loop):
            self._loop.run_until_complete(task)
        self._loop.close()

    async def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    # TODO: decouple scheduling and running the job
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        # TODO: Assert correct type of handler?
        async def do():
            try:
                job.status = JobStatus.running
                await job.handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb)
            except Exception as e:
                on_log_message_cb(str(e))
                job.status = JobStatus.failed
                logger.exception(f"Job {job.id} failed.")

        asyncio.run_coroutine_threadsafe(do(), self._loop)

    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        pass

    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        pass

    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        pass


class _KFPSchedulerBackendBase(_NaiveSchedulerBackend):
    def __init__(self, to_artifacts: Callable[[Any], list[JobArtifact]]):
        super().__init__()
        self._to_artifacts = to_artifacts


class _KFPLocalSchedulerBackend(_KFPSchedulerBackendBase):
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        async def do():
            from kfp import local

            local.init(runner=local.SubprocessRunner(use_venv=False))

            job.status = JobStatus.running
            try:
                artifacts = self._to_artifacts(job.handler().output)
                for artifact in artifacts:
                    on_artifact_collected_cb(artifact)

                job.status = JobStatus.completed
            # TODO: Confirm failures in pipelines are caught
            except Exception as e:
                job.status = JobStatus.failed
                on_log_message_cb(str(e))
                logger.exception(f"Job {job.id} failed.")

        asyncio.run_coroutine_threadsafe(do(), self._loop)


class _KFPRemoteSchedulerBackend(_KFPSchedulerBackendBase):

    # stolen from: https://github.com/meta-llama/llama-stack/compare/main...cdoern:llama-stack:ilab-dsp
    # TODO: confirm the source of the code
    # TODO: check if all this code is really needed - doesn't kfp library provide a simpler interface?
    @staticmethod
    def get_kfp_client():
        from kfp import Client
        from kubernetes.client import CustomObjectsApi
        from kubernetes.client.configuration import Configuration
        from kubernetes.client.exceptions import ApiException
        from kubernetes.config import list_kube_config_contexts
        from kubernetes.config.config_exception import ConfigException
        from kubernetes.config.kube_config import load_kube_config


        config = Configuration()
        try:
            load_kube_config(client_configuration=config)
            token = config.api_key["authorization"].split(" ")[-1]
        except (KeyError, ConfigException) as e:
            raise ApiException(
                401, "Unauthorized, try running `oc login` command first"
            ) from e
        Configuration.set_default(config)

        _, active_context = list_kube_config_contexts()
        namespace = active_context["context"]["namespace"]

        # TODO: this is not really kfp, is it? do we have to deal with ocp vs k8s here?
        dspas = CustomObjectsApi().list_namespaced_custom_object(
            "datasciencepipelinesapplications.opendatahub.io",
            "v1alpha1",
            namespace,
            "datasciencepipelinesapplications",
        )

        try:
            dspa = dspas["items"][0]
        except IndexError as e:
            raise ApiException(404, "DataSciencePipelines resource not found") from e

        try:
            if dspa["spec"]["dspVersion"] != "v2":
                raise KeyError
        except KeyError as e:
            raise EnvironmentError(
                "Installed version of Kubeflow Pipelines does not meet minimal version criteria. Use KFPv2 please."
            ) from e

        try:
            host = dspa["status"]["components"]["apiServer"]["externalUrl"]
        except KeyError as e:
            raise ApiException(
                409,
                "DataSciencePipelines resource is not ready. Check for .status.components.apiServer",
            ) from e

        with warnings.catch_warnings(action="ignore"):
            return Client(existing_token=token, host=host)

    # TODO: move error handling for local and remote cases into base class, if possible?
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        async def do():
            client = self.get_kfp_client()
            res = client.create_run_from_pipeline_func(
                pipeline_func=job.handler,
                run_name=job.id,
            )
            job.provider_id = res.run_id
            job.status = JobStatus.running

            # TODO: extract artifacts... is it possible via REST API?
            # See: https://github.com/kubeflow/pipelines/issues/9858
            while not job.completed_at:
                run = client.get_run(job.provider_id)

                # Revisit the map between states in KFP and LLS
                match run.state:
                    # TODO: are there enums for the states?
                    case "SUCCEEDED":
                        on_status_change_cb(JobStatus.completed)
                    case "FAILED" | "CANCELED" | "SKIPPED":
                        on_status_change_cb(JobStatus.failed)
                    case "RUNNING":
                        logger.info(f"Job {job.id} (kfp: {job.provider_id}) is still running")
                    case _:
                        logger.warning(f"Unhandled run state: {run.state}")

                # TODO: is there a better way to wait for the job to finish without polling?
                await asyncio.sleep(5)

        asyncio.run_coroutine_threadsafe(do(), self._loop)


_BACKENDS = {
    "naive": _NaiveSchedulerBackend,
    "kfp-local": _KFPLocalSchedulerBackend,
    "kfp-remote": _KFPRemoteSchedulerBackend,
}


def _get_backend_impl(backend: str, to_artifacts: Callable[[Any], list[JobArtifact]] | None = None) -> _SchedulerBackend:
    try:
        if to_artifacts is not None:
            return _BACKENDS[backend](to_artifacts=to_artifacts)
        return _BACKENDS[backend]()
    except KeyError as e:
        raise ValueError(f"Unknown backend {backend}") from e


class Scheduler:
    def __init__(self, backend: str = "naive", to_artifacts: Callable[[Any], list[JobArtifact]] | None = None):
        # TODO: if server crashes, job states are lost; we need to persist jobs on disc
        self._jobs: dict[JobID, Job] = {}
        self._backend = _get_backend_impl(backend, to_artifacts)

    def _on_log_message_cb(self, job: Job, message: str) -> None:
        msg = (datetime.now(timezone.utc), message)
        # At least for the time being, until there's a better way to expose
        # logs to users, log messages on console
        logger.info(f"Job {job.id}: {message}")
        job.append_log(msg)
        self._backend.on_log_message_cb(job, msg)

    def _on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        self._on_log_message_cb(job, f"Job status changed from {job.status} to {status}")
        job.status = status
        self._backend.on_status_change_cb(job, status)

    def _on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        job.register_artifact(artifact)
        self._backend.on_artifact_collected_cb(job, artifact)

    def schedule(self, type_: JobType, job_id: JobID, handler: JobHandler) -> JobID:
        job = Job(type_, job_id, handler)
        if job.id in self._jobs:
            raise ValueError(f"Job {job.id} already exists")

        self._jobs[job.id] = job
        job.status = JobStatus.scheduled
        self._backend.schedule(
            job,
            functools.partial(self._on_log_message_cb, job),
            functools.partial(self._on_status_change_cb, job),
            functools.partial(self._on_artifact_collected_cb, job),
        )

        return job.id

    def cancel(self, job_id: JobID) -> None:
        self.get_job(job_id).cancel()

    def get_job(self, job_id: JobID) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as e:
            raise ValueError(f"Job {job_id} not found") from e

    def get_jobs(self, type_: JobType | None = None) -> list[Job]:
        jobs = list(self._jobs.values())
        if type_:
            jobs = [job for job in jobs if job._type == type_]
        return jobs

    async def shutdown(self):
        # TODO: also cancel jobs once implemented
        await self._backend.shutdown()
