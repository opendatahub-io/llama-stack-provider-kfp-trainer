#!/bin/sh

podman tag localhost/llama-stack-provider-kfp-trainer quay.io/booxter/llama-stack-provider-kfp-trainer:latest
podman push quay.io/booxter/llama-stack-provider-kfp-trainer:latest
