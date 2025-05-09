from pathlib import Path
import tempfile

from moto import mock_aws
import pytest

from llama_stack_provider_kfp_trainer.s3 import Client


def s3_client():
    return Client(
        bucket="test-bucket",
        region_name="us-east-1",
    ).create_bucket()


@mock_aws
def test_upload_download():
    c = s3_client()

    # Upload
    data = b"testdata"
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        c.upload(Path(tmp.name), "test-object")

    # Download
    with tempfile.NamedTemporaryFile() as tmp:
        c.download("test-object", Path(tmp.name), override=True)
        tmp.seek(0)
        assert tmp.read() == data


@mock_aws
def test_download_fail_no_destdir():
    c = s3_client()
    with pytest.raises(FileNotFoundError):
        c.download("test-object", Path("/nonexistent/test-object"))


@mock_aws
def test_download_fail_file_exists():
    c = s3_client()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"")
        tmp.flush()
        with pytest.raises(FileExistsError):
            c.download("test-object", Path(tmp.name))
