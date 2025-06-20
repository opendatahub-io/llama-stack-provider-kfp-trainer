# llama-stack-provider-kfp-trainer

Llama Stack Post Training Provider using KubeFlow Pipelines.

This provider demonstrates how the same KFP pipeline using `torchtune` can be
define the training workfload for both local and remote execution.


## Prepare virtual environment

```
./scripts/prepare-venv.sh
```

## Run llama-stack server (local)

```
./scripts/run-local-server.sh
```

## Run llama-stack server (remote)

```
./scripts/run-remote-server.sh
```

## Prepare model files

The model should be present on the system. For `local` mode, use `llama stack
download` to download llama model checkpoints.

If using `remote`, the downloaded model should then be pushed to s3 bucket. The
following script assumes that AWS credentials are configured in the
environment.

```
./scripts/upload-model.py
```

## Train

```
./scripts/run-training.sh
```

Depending on the provider configuration, training will then execute either
locally via KFP Local subprocess runner or remotely on DSP service.

## Implementation

You can find some development notes in the [docs](docs/) directory.
