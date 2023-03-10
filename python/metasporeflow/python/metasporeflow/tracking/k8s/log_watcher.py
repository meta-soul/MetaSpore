import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from uploader.local_uploader import LocalUploader
from uploader.obs_uploader import OBSUploader
from uploader.s3_uploader import S3Uploader
from uploader.upload_type import UploadType
import os


class LogWatcher:
    def __init__(self, path):
        self.observer = Observer()
        self.path = path

    def run(self):
        event_handler = self.Handler()
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Error")
        self.observer.join()

    class Handler(PatternMatchingEventHandler):
        def __init__(self):
            super().__init__(
                patterns=["tracking.log.*"],
                ignore_patterns=["tracking.log"],
                ignore_directories=True,
                case_sensitive=False,
            )


        def on_any_event(self, event):
            upload_type = os.environ.get("UPLOAD_TYPE", "LOCAL")
            print("Upload type: {}".format(upload_type))

            if event.event_type == "moved":
                print("[{}] noticed: [{}] on: [{}] ".format(
                    time.asctime(), event.event_type, event.dest_path))
                src_path = event.dest_path
                if upload_type == UploadType.LOCAL.value:
                    uploader = LocalUploader(src_path)
                elif upload_type == UploadType.OBS.value:
                    uploader = OBSUploader(src_path)
                elif upload_type == UploadType.S3.value:
                    uploader = S3Uploader(src_path)
                else:
                    raise Exception("Unsupported upload type: %s" % upload_type)

                uploader.upload()


if __name__ == '__main__':
    print("Upload type: {}".format(os.environ.get("UPLOAD_TYPE")))
    path = os.getcwd()
    print("LogWatcher listen path : {}".format(path))
    w = LogWatcher(path)
    w.run()
