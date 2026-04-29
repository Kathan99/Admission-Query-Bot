import os
import boto3
from botocore.exceptions import ClientError

try:
    from backend.config import settings
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings


def _get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
        region_name=settings.aws_region,
    )


def upload_file_to_s3(local_path: str, s3_key: str) -> bool:
    """Upload a local file to S3. Returns True on success."""
    try:
        client = _get_s3_client()
        client.upload_file(local_path, settings.s3_bucket, s3_key)
        print(f"Uploaded {local_path} → s3://{settings.s3_bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"S3 upload failed for {s3_key}: {e}")
        return False


def upload_bytes_to_s3(data: bytes, s3_key: str) -> bool:
    """Upload raw bytes to S3. Returns True on success."""
    try:
        client = _get_s3_client()
        client.put_object(Body=data, Bucket=settings.s3_bucket, Key=s3_key)
        print(f"Uploaded bytes → s3://{settings.s3_bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"S3 upload (bytes) failed for {s3_key}: {e}")
        return False


def download_file_from_s3(s3_key: str, local_path: str) -> bool:
    """Download an S3 object to a local path. Returns True on success."""
    try:
        client = _get_s3_client()
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        client.download_file(settings.s3_bucket, s3_key, local_path)
        return True
    except ClientError as e:
        print(f"S3 download failed for {s3_key}: {e}")
        return False


def sync_s3_to_local(local_dir: str, prefix: str = "") -> int:
    """Download all objects under `prefix` in S3 into `local_dir`.

    Existing local files are skipped when their size already matches the S3
    object size (cheap idempotency check — avoids re-downloading unchanged PDFs
    on every ingestion run).

    Returns the number of files newly downloaded.
    """
    try:
        client = _get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix)
        os.makedirs(local_dir, exist_ok=True)
        count = 0
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):].lstrip("/")
                if not rel:
                    continue
                local_path = os.path.join(local_dir, rel)
                # Skip if local copy already matches S3 size
                if os.path.exists(local_path) and os.path.getsize(local_path) == obj["Size"]:
                    continue
                os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                client.download_file(settings.s3_bucket, key, local_path)
                count += 1
        print(f"S3 sync: downloaded {count} new file(s) from s3://{settings.s3_bucket}/{prefix} → {local_dir}")
        return count
    except ClientError as e:
        print(f"S3 sync failed: {e}")
        return 0


def list_s3_files(prefix: str = "") -> list[str]:
    """List all object keys under `prefix` in the configured S3 bucket."""
    try:
        client = _get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix)
        keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
    except ClientError as e:
        print(f"S3 list failed: {e}")
        return []
