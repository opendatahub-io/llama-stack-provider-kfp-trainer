#!/bin/sh

# This script is used to create a virtual environment for the Llama Stack KFP Trainer

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Should probably be in the default dependencies for llama-stack but is not -
# the reference provider is loaded when no telemetry section is configured
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# This dependency doesn't seem to be tracked at all
pip install aiosqlite

# These come from external provider dependencies
pip install torch torchtune==0.5.0 torchao==0.8.0 numpy kfp kubernetes
