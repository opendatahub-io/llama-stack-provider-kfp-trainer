## Dependencies

This project relies on the external providers support in llama-stack, which is
expected to be released in 0.2.2, see [here](https://github.com/meta-llama/llama-stack/releases).

This provider extends the in-tree `TorchtunePostTrainingImpl` inline provider.
This is not meant to be a pattern to use in case this approach is used for
production needs. This approach is to limit the scope of changes and better
showcase Kubeflow Pipelines. It could as well be HFTrainer or any other
provider.

This provider also uses the `scheduler` module to run `async` tasks. It extends
it with additional features. Specifically, two new KFP backend implementations
are added: one to use
[local runner](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/)
and another to use KFP remote API.


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
located at `~/.llama/checkpoints/Llama3.2-3b-instruct.tar.gz` that contains
flat model files with Meta model. You can use `./scripts/upload-model.py` to
create the tarball file. (It will also attempt to upload the file to remote S3
bucket, for remote mode.)

When SFT is triggered, the provider imports the local tarball file as an
artifact and execute the pipeline. The pipeline contains a single component
that runs the `llama-stack` LoRA training with `torchtune` named `train`. The
result of the component execution is returned as the pipeline output.

Note: KFP local doesn't support remote importers, which is why we can't reuse
the S3 import when running the pipeline locally. This may change
[in the future](https://github.com/kubeflow/pipelines/pull/11875).


### Remote Pipelines

The following preconditions are required to run the remote pipelines:

- The user must have access to the KFP server. Its `kubectl` context
  should be set to the cluster where the KFP server is running. In OpenShift,
  it means deploying Data Science Pipelines using the appropriate operator.
- The user must have access to the S3 bucket where the model artifacts are
  stored. This bucket is also used to pass dataset artifacts between
  `llama-stack` server and KFP components. This is the same bucket that was
  used when configuring the KFP server. The bucket can be configured in
  `run.yaml` file or by passing `KFP_S3_BUCKET` environment variable.

When the `mode` is Remote, `kubernetes` KFP client is used instead to run
SFT jobs. Instead of running the pipeline using the local runner, the pipeline
is compiled into YAML IR and submitted to the KFP server.

For the model artifact, the code assumes that it's located at
`s3://$KFP_S3_BUCKET/<MODEL_NAME>.tar.gz`. You can use
`./scripts/upload-model.py` to pack and upload a model to the S3 bucket.

Note: KFP servers do not support non-static AWS tokens. Please ask your
administrator to configure static credentials with access to the desired S3
bucket.

Since the remote pipeline doesn't have access to dependencies installed with
`llama-stack`, they are defined (copied) to the component decorator as
`packages_to_install` argument. These packages are installed when spinning the
component container (at the time of writing, we are using a custom container
build `quay.io/booxter/llama-stack-provider-kfp-trainer:latest` that includes
`torchtune` dependencies in hope of speeding up job execution. You can also use
something more generic like `quay.io/fedora/python-311` if needed).


## Datasets

Datasets are referred from LLS API and are extracted by the `pipeline`
function, then uploaded to the S3 bucket - or local file system in case of
Local mode. When the pipeline is executed, it fetches and extracts the dataset
to local filesystem and passes the result as an artifact.


## Acceleration

The `torchtune` provider supports CUDA. If CUDA is not available, the CPU mode
is used instead.

If acceleration is used, the user must have access to the GPU nodes in the
cluster. If so, the `USE_GPU=True` variable can be set to enable acceleration;
alternatively, set `use_gpu` key in the `run.yaml` file. Note: at the time of
writing, acceleration uses the hardcoded `nvidia.com/gpu` resource name. If
alternative acceleration resources are needed, the provider code will have to
be patched. (Note that this may require modifications to the `torchtune`
provider as well since it assumes CUDA environment at the moment.)

Note: At the time of writing, `torchtune` provider will always use a CUDA
device if available, even if the `USE_GPU` variable is not set. The variable
will only affect `kubernetes` resource requests and has no effect for the local
mode. Fixing it would require introducing a concept of resource requests in
`llama-stack` server implementation (namely, the job scheduler module) as well
as exposing resource management via API.
