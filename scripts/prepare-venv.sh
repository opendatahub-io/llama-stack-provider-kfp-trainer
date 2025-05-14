#!/bin/sh

# This script is used to create a virtual environment for the Llama Stack KFP Trainer

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip

# Should probably be in the default dependencies for llama-stack but is not -
# the reference provider is loaded when no telemetry section is configured
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# This dependency doesn't seem to be tracked at all
pip install aiosqlite
pip install .
