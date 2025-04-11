# llama-stack-provider-kfp-trainer

Llama Stack Post Training Provider using KubeFlow Pipelines.

This provider demonstrates how the same KFP pipeline using `torchtune` can be
define the training workfload for both local and remote execution.


## Prepare environment

```
./scripts/prepare-venv.sh
```

## Run llama-stack server

```
./scripts/run-server.sh
```

## Train

```
./scripts/run-training.sh
```
