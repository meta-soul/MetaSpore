from .base_uploader import BaseUploader


class LocalUploader(BaseUploader):
    def __init__(self, src_path):
        super(LocalUploader, self).__init__(src_path)

    def upload(self):
        import shutil
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Uploading to local: %s" % self.src_path)
        shutil.copyfile(self.src_path, self.upload_path)
        logger.info("Upload to local successfully: %s" % self.upload_path)
        return self.upload_path
