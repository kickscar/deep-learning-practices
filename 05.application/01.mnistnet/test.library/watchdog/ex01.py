import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

from watchdog.events import FileSystemEventHandler, LoggingEventHandler
from watchdog.observers import Observer


# class FileModifiedEventHandler(LoggingEventHandler):
class FileModifiedEventHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = datetime.now()

    def on_modified(self, event):
        # if datetime.now() - self.last_modified < timedelta(seconds=1):
        #   return

        # self.last_modified = datetime.now()

        # if event.is_directory:
        #    return

        print(f'Event type: {event.event_type}  path : {event.src_path}')
        print(event.is_directory)

    # def on_created(self, event):
    #     print('created')
    #
    # def on_moved(self, event):
    #     print('moved')
    #
    # def on_deleted(self, event):
    #     print('deleted')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file = os.path.join(Path(os.getcwd()).parent.parent, 'images')
print(file)

observer = Observer()
observer.schedule(FileModifiedEventHandler(), file, recursive=True)
observer.start()


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
