#!/bin/bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-llama-stack-provider-kfp-trainer:latest}"
CONTAINERFILE="${CONTAINERFILE:-Containerfile}"

if [[ ! -f "$CONTAINERFILE" ]]; then
  echo "Error: $CONTAINERFILE not found in current directory."
  exit 1
fi

echo "Building container image: $IMAGE_NAME"
podman build --arch=amd64 -f "$CONTAINERFILE" -t "$IMAGE_NAME" .

echo "Build complete: $IMAGE_NAME"
