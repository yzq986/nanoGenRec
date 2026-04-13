"""S3 upload/download/parse utilities."""

import os


def parse_s3_path(s3_path: str):
    """Parse s3://bucket/key into (bucket, key)."""
    s3_path_clean = s3_path.replace('s3://', '')
    bucket = s3_path_clean.split('/')[0]
    key = '/'.join(s3_path_clean.split('/')[1:])
    return bucket, key


def upload_to_s3(local_path: str, s3_path: str):
    """Upload file to S3."""
    import boto3
    s3 = boto3.client('s3')
    bucket, key = parse_s3_path(s3_path)
    s3.upload_file(local_path, bucket, key, ExtraArgs={'ACL': 'bucket-owner-full-control'})
    print(f"Uploaded to s3://{bucket}/{key}")


def download_from_s3(s3_path: str, local_path: str):
    """Download file from S3."""
    import boto3
    print(f"Downloading {s3_path} -> {local_path}")
    s3 = boto3.client('s3')
    bucket, key = parse_s3_path(s3_path)
    s3.download_file(bucket, key, local_path)
    print(f"  Done: {os.path.getsize(local_path) / 1024 / 1024:.1f} MB")
