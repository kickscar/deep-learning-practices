import time
import logging
from watchdog.events import FileSystemEventHandler, LoggingEventHandler
from watchdog.observers import Observer


# class FileModifiedEventHandler(FileSystemEventHandler):
class FileModifiedEventHandler(LoggingEventHandler):
    def on_created(self, event):
        print('created')

    def on_modified(self, event):
        super(FileModifiedEventHandler, self).on_modified(event)
        print('modified')

    def on_moved(self, event):
        print('moved')

    def on_deleted(self, event):
        print('deleted')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

observer = Observer()
observer.schedule(FileModifiedEventHandler(), '../../images/test.bmp', recursive=True)
observer.start()


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
