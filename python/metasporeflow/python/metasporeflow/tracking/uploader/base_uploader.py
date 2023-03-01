from metasporeflow.tracking.upload_type import UploadType


class BaseUploader(object):
    def __init__(self, src_path):
        self.src_path = src_path
        self.upload_path_local = "/tmp/tracking/"
        self.upload_type = None
        self.upload_path = None
        self.access_key = None
        self.secret_key = None
        self.endpoint = None
        self._set_upload_config()

    def upload(self):
        raise NotImplementedError

    def _set_upload_config(self):
        import os
        upload_type = os.environ.get("UPLOAD_TYPE", "LOCAL")
        upload_path = os.environ.get("UPLOAD_PATH")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint = os.environ.get("AWS_ENDPOINT")

        if upload_type not in UploadType.__members__:
            raise Exception("Unsupported upload type: %s" % self.upload_type)

        if upload_type == UploadType.LOCAL.value:
            if upload_path is None:
                upload_path = self.upload_path_local
            if not os.path.exists(upload_path):
                os.makedirs(upload_path)
        else:
            if access_key is not None and secret_key is not None and endpoint is not None and upload_path is not None:
                self.access_key = access_key
                self.secret_key = secret_key
                self.endpoint = endpoint
            else:
                raise Exception("ENV AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT and UPLOAD_PATH are required")

        self.upload_type = upload_type
        self.upload_path = upload_path

    @property
    def bucket_name(self):
        from urllib.parse import urlparse
        results = urlparse(self.upload_path, allow_fragments=False)
        bucket = results.netloc
        return bucket

    @property
    def _file_name(self):
        import os
        return os.path.basename(self.src_path)

    @property
    def object_key(self):
        from urllib.parse import urlparse
        results = urlparse(self.upload_path, allow_fragments=False)
        object_key = results.path.lstrip('/')
        return object_key

    @property
    def region(self):
        import re
        if self.upload_type == UploadType.S3.value:
            pattern = r's3\.([A-Za-z0-9\-]+?)\.amazonaws\.com(\.cn)?$'
        elif self.upload_type == UploadType.OBS.value:
            pattern = r'obs\.([A-Za-z0-9\-]+?)\.myhuaweicloud\.com$'
        match = re.match(pattern, self.endpoint)
        if match is None:
            message = 'invalid s3 endpoint %r' % self.endpoint
            raise RuntimeError(message)
        aws_region = match.group(1)
        return aws_region
