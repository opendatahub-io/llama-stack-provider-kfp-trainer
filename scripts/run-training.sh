#!/bin/sh

python -m venv .venv-client
. .venv-client/bin/activate
pip install llama-stack-client

# Client fails to pull some dependencies that it imports
pip install fire requests

pip install colorama

export INFERENCE_MODEL=Llama-3.2-3B-Instruct
python train.py
