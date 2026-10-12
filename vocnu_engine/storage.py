"""Object storage: Cloudflare R2 in production, a local folder for CLI runs.

The worker only reads `uploads/...` and only writes `jobs/<job_id>/...` (worker-contract.md).
"""
from pathlib import Path
import re
import shutil

from .config import secret
from .errors import WorkerError

UPLOAD_KEY = re.compile(r'uploads/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[^/\x00-\x1f]{1,255}')
JOB_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]{7,63}')


def check_upload_key(key):
    if not isinstance(key, str) or '..' in key or not UPLOAD_KEY.fullmatch(key):
        raise WorkerError('not_found', 'The uploaded file reference is invalid.')
    return key


def check_job_id(job_id):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError('Invalid job_id')
    return job_id


def output_key(job_id, name):
    return f'jobs/{check_job_id(job_id)}/{name}'


class R2:
    def __init__(self, client=None, bucket=None):
        self.bucket = bucket or secret('R2_BUCKET')
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client(
                's3', endpoint_url=secret('R2_ENDPOINT'), region_name='auto',
                aws_access_key_id=secret('R2_ACCESS_KEY_ID'), aws_secret_access_key=secret('R2_SECRET_ACCESS_KEY'),
                config=Config(signature_version='s3v4', retries={'max_attempts': 5, 'mode': 'standard'},
                              connect_timeout=10, read_timeout=120))
        self.client = client

    def size(self, key):
        from botocore.exceptions import ClientError
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=key)['ContentLength'])
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
                return None
            raise

    def presign_get(self, key, expires=900):
        return self.client.generate_presigned_url('get_object', Params={'Bucket': self.bucket, 'Key': key},
                                                  ExpiresIn=expires)

    def download(self, key, path):
        if self.size(key) is None:
            raise WorkerError('not_found', 'The uploaded file was not found. It may have expired; upload it again.')
        self.client.download_file(self.bucket, key, str(path))
        return Path(path)

    def upload(self, path, key, content_type):
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={'ContentType': content_type})
        return key

    def delete(self, key):
        """Removes an object; a missing one is fine (S3 deletes are idempotent)."""
        self.client.delete_object(Bucket=self.bucket, Key=key)


class LocalStore:
    """Writes outputs below `root` with the same keys; reads sources as plain paths."""

    def __init__(self, root):
        self.root = Path(root)

    def upload(self, path, key, content_type):
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return key

    def delete(self, key):
        (self.root / key).unlink(missing_ok=True)
