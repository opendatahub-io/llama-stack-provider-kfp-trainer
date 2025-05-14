#!/bin/sh

python -m venv .venv-client
. .venv-client/bin/activate
pip install llama-stack-client

export INFERENCE_MODEL=Llama-3.2-3B-Instruct
python train.py
