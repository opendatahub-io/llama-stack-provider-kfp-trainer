import os
from pathlib import Path
from typing import Self

import botocore.exceptions
import boto3
# from botocore.client import Config


class Client:
    def __init__(
        self,
        bucket: str,
        region_name: str | None = None,
    ):
        self.bucket = bucket
        self.conn = boto3.resource("s3", region_name=region_name)

    def upload(self, src: Path, object_name: str | None = None) -> Self:
        if object_name is None:
            object_name = str(src.name)

        if not src.is_file():
            raise FileNotFoundError(f"File {src} does not exist.")

        if not os.access(src, os.R_OK):
            raise PermissionError(f"File {src} is not readable.")

        try:
            self.conn.Bucket(self.bucket).upload_file(str(src), object_name)
        except Exception as e:
            raise RuntimeError(f"Failed to upload {src} to bucket {self.bucket}: {e}")
        return self

    def download(self, object_name: str, dest: Path, override: bool = False) -> Self:
        if not dest.parent.is_dir():
            raise FileNotFoundError(f"Directory {dest.parent} does not exist.")

        if dest.is_file() and not override:
            raise FileExistsError(
                f"File {dest} already exists. Use override=True to overwrite."
            )

        try:
            # TODO: stream, don't cache the whole file in memory
            body = self.conn.Object(self.bucket, object_name).get()["Body"].read()
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {object_name} from bucket {self.bucket}: {e}"
            )

        with open(dest, "wb") as f:
            f.write(body)
        return self

    def create_bucket(self) -> Self:
        try:
            self.conn.create_bucket(Bucket=self.bucket)
        except Exception as e:
            raise RuntimeError(f"Failed to create bucket {self.bucket}: {e}")
        return self

    def exists(self, object_name: str) -> bool:
        try:
            self.conn.ObjectSummary(self.bucket, object_name).load()
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise RuntimeError(
                f"Failed to check existence of {object_name} in bucket {self.bucket}: {e}"
            )
