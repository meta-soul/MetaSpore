from .base_uploader import BaseUploader


class S3Uploader(BaseUploader):
    def __init__(self, src_path):
        super(S3Uploader, self).__init__(src_path)

    def upload(self):
        import boto3
        import os
        import logging
        from botocore.exceptions import ClientError
        logger = logging.getLogger(__name__)
        logger.info("Uploading to S3: %s" % self.src_path)
        # Create a S3Client instance
        s3_client = boto3.client('s3', self.region)
        # Get file size
        file_size = os.path.getsize(self.src_path)
        # If file size is less than 5M, use put object; otherwise, use multipart upload
        if file_size < 5 * 1024 * 1024:
            try:
                response = s3_client.upload_file(self.src_path, self.bucket_name, self.object_key)
            except ClientError as e:
                logging.error(e)
                return False
            return True
