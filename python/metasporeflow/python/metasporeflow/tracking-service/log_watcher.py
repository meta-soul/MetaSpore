from utils.log_util import get_logger

def watch_log():
    import os
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    print("watching log dir: "+os.getcwd())
    class MyHandler(FileSystemEventHandler):
        def __init__(self, logger):
            self.logger = logger

        def on_modified(self, event):
            if event.src_path.endswith('.log'):
                self.logger.info('file modified: %s' % event.src_path)
    event_handler = MyHandler(get_logger())
    observer = Observer()
    observer.schedule(event_handler, path=os.getcwd(), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    watch_log()