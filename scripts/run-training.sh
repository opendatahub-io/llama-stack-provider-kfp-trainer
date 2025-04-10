#!/bin/sh

python -m venv .venv-client
source .venv-client/bin/activate
pip install llama-stack-client

export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
python train.py
