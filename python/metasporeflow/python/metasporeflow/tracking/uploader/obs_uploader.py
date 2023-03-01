from .base_uploader import BaseUploader


class OBSUploader(BaseUploader):
    def __init__(self, src_path):
        super(OBSUploader, self).__init__(src_path)

    def upload(self):
        from obs import ObsClient
        import os
        from obs import PutObjectHeader
        headers = PutObjectHeader()
        headers.contentType = 'text/plain'
        # Create a ObsClient instance
        obsClient = ObsClient(access_key_id=self.access_key,
                              secret_access_key=self.secret_key,
                              server=self.endpoint)
        # Get file size
        file_size = os.path.getsize(self.src_path)
        # If file size is less than 10M, use put object; otherwise, use multipart upload
        partSize = 10 * 1024 * 1024
        if file_size < partSize:
            obsClient.putFile(bucketName=self.bucket_name,
                                    objectKey=self.object_key,
                                    file_path=self.src_path,
                                    headers=headers)
        else:
            taskNum = 5
            enableCheckpoint = True
            uploadFile = 'localfile'
            try:
                resp = obsClient.uploadFile(self.bucket_name, self.object_key, uploadFile,
                                            partSize, taskNum, enableCheckpoint)
                if resp.status < 300:
                    print('requestId:', resp.requestId)
                else:
                    print('errorCode:', resp.errorCode)
                    print('errorMessage:', resp.errorMessage)
            except:
                import traceback
                print(traceback.format_exc())