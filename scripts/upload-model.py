#!/usr/bin/env python3

from argparse import ArgumentParser
import sys
from pathlib import Path
import tarfile

from llama_stack_provider_kfp_trainer.s3 import Client


# Parse arguments with argparse
parser = ArgumentParser(description="Upload model to S3 bucket.")
parser.add_argument(
    "--bucket",
    type=str,
    default="rhods-dsp-dev",
    help="S3 bucket name",
)
parser.add_argument(
    "--model",
    type=str,
    default="Llama3.2-3B-Instruct",
    help="Model name",
)
args = parser.parse_args()

# Validate the model directory
checkpoints_dir = Path.home() / ".llama" / "checkpoints"
model_path = checkpoints_dir / args.model
if not model_path.is_dir():
    print(f"Model directory {model_path} does not exist.")
    sys.exit(1)

expected_files = [
    model_path / fname
    for fname in (
        "checklist.chk",
        "consolidated.00.pth",
        "params.json",
        "tokenizer.model",
    )
]
for file in expected_files:
    if not (file).is_file():
        print(f"File {file} does not exist.")
        sys.exit(1)

# Check that the model is not already uploaded
s3_client = Client(args.bucket)

object_name = f"{args.model}.tar.gz"
if s3_client.exists(object_name):
    print(f"Model {args.model} already exists in S3 bucket.")
    sys.exit(0)

print("Model not found in S3 bucket, proceeding...")

# Create a tarball of the model directory
tarball_path = Path(f"{model_path}.tar.gz")
print(f"Creating tarball {tarball_path}...")

with tarfile.open(tarball_path, "w:gz") as tar:
    for file in expected_files:
        tar.add(file, arcname=file.name)

# Upload the tarball to S3
print(f"Uploading model {args.model}...")
s3_client.upload(tarball_path, object_name)

# Confirm the upload was successful
if not s3_client.exists(object_name):
    print(f"Failed to upload model {args.model} to S3 bucket {args.bucket}.")
    sys.exit(1)
