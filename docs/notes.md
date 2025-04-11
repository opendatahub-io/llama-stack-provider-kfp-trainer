## Dependencies

This project relies on the external providers support in llama-stack, which is
expected to be released in 0.2.2, see [here](https://github.com/meta-llama/llama-stack/releases).

This provider extends the in-tree `TorchtunePostTrainingImpl` inline provider.
This is not meant to be a pattern to use in case this approach is used for
production needs. This approach is to limit the scope of changes and better
showcase Kubeflow Pipelines.

This provider also uses the `scheduler` module to run `async` tasks. This
module is proposed in upstream
[here](https://github.com/meta-llama/llama-stack/pull/1437). Once it's merged
in upstream, this provider will switch to use it.


## Execution flow

1. The user submits a training job using the `llama-stack` CLI.
2. The `llama-stack` server sends the request to `TorchtuneKFPPostTrainingImpl`
   provider.
3. The provider passes LLS API and other arguments to `pipeline` function
   defined in `pipeline.py`. One of the arguments is `mode` which defines
   whether the pipeline should be executed locally or remotely.


### Local Pipelines

When the `mode` is Local, `SubprocessRunner` for [KFP local](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/)
is used. This runner creates a new process that reuses the same Python
environment as `llama-stack` (`use_venv=False` is used).

The code assumes that the user running `llama-stack` has a particular tarball
located at `~/llama3.2-3b-instruct.tar.gz` that contains flat model files with
Meta model. (Note: This file is a pure `tar` file, without `gzip` compression.
This is a bug to be fixed.)

The pipeline will import the local tarball file as an artifact and execute the
pipeline. The pipeline contains a single component that runs the `llama-stack`
LoRA training with `torchtune`. The result is returned as the pipeline output.

Note: KFP local doesn't support remote importers, which is why we can't reuse
the s3 import when running the pipeline locally.


### Remote Pipelines

When the `mode` is Remote, `kubernetes` KFP client is used instead.

To access the KFP server, the session token is used. The session token can be
initialized with `oc login` command as usual.

For the model artifact, the code assumes that it's located at
`s3://rhods-dsp-dev/llama3.2-3b-instruct.tar.gz`.

Since the remote pipeline doesn't have access to dependencies installed with
`llama-stack`, they are defined (copied) to the component decorator as
`packages_to_install` argument. These packages are installed when spinning the
component container (currently based on `quay.io/fedora/python-311` image). We
may want to pre-build a container with necessary dependencies if we'd like to
speed up initialization.


## Datasets

Datasets are referred from LLS API and are extracted by the `pipeline`
function, then passed as an argument to the pipeline. Because of limits to the
size of the pipeline, we currently truncate the dataset to just first 10 rows.
We should switch to passing an actual Artifact to the pipeline to fix this
limitation.


## Acceleration

The `torchtune` provider supports CUDA. If CUDA is not available, the CPU mode
is used instead.
