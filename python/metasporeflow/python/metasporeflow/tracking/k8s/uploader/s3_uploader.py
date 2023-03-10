from .base_uploader import BaseUploader


class S3Uploader(BaseUploader):
    def __init__(self, src_path):
        super(S3Uploader, self).__init__(src_path)

    def upload(self):
        import subprocess
        args = ['aws', '--endpoint-url', self.endpoint_url, 's3', 'cp', self.src_path, self.upload_path]
        subprocess.run(args, input=b'', check=True)

    @property
    def endpoint_url(self):
        endpoint = self.endpoint
        if endpoint.startswith('http://') or endpoint.startswith('https://'):
            s3_endpoint = endpoint
        else:
            s3_endpoint = 'https://' + endpoint
        return s3_endpoint

    def _set_S3_env(self):
        import os
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        os.environ["AWS_DEFAULT_REGION"] = self.region
