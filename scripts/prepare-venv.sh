#!/bin/sh

# This script is used to create a virtual environment for the Llama Stack KFP Trainer

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip

# Should probably be in the default dependencies for llama-stack but is not -
# the reference provider is loaded when no telemetry section is configured
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# Install some dependencies not pulled by llama-stack for some reason
pip install aiosqlite fastapi uvicorn

pip install -e .
