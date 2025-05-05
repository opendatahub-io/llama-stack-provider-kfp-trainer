# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import warnings
from typing import Any, Callable, TypeAlias

from llama_stack.log import get_logger
from llama_stack.providers.utils import scheduler

logger = get_logger(name=__name__, category="scheduler")


JobArtifact: TypeAlias = scheduler.JobArtifact
JobStatus: TypeAlias = scheduler.JobStatus
JobID: TypeAlias = scheduler.JobID
Job: TypeAlias = scheduler.Job


# TODO: introduce a base backend class upstream that we could officially extend here
class _KFPSchedulerBackendBase(scheduler._NaiveSchedulerBackend):
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


# TODO: introduce a backend registration hook in upstream
scheduler._BACKENDS["kfp-local"] = _KFPLocalSchedulerBackend
scheduler._BACKENDS["kfp-remote"] = _KFPRemoteSchedulerBackend


def _get_backend_impl(backend: str, to_artifacts: Callable[[Any], list[JobArtifact]] | None = None):
    try:
        if to_artifacts is not None:
            return scheduler._BACKENDS[backend](to_artifacts=to_artifacts)
        return scheduler._BACKENDS[backend]()
    except KeyError as e:
        raise ValueError(f"Unknown backend {backend}") from e


class Scheduler(scheduler.Scheduler):
    def __init__(self, backend: str = "naive", to_artifacts: Callable[[Any], list[JobArtifact]] | None = None):
        # TODO: if server crashes, job states are lost; we need to persist jobs on disc
        self._jobs: dict[JobID, Job] = {}
        self._backend = _get_backend_impl(backend, to_artifacts)
