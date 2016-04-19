import logging
from logging.handlers import RotatingFileHandler
import os
import time

__author__ = 'Deyang'


def get_logger(name):
    logger = logging.getLogger(name)
    file_handler = get_file_handler(name)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger


def get_file_handler(name):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(dirpath, '..', '%s.log' % name)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=100000000, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(pathname)s@%(lineno)d]: %(message)s'))
    return file_handler


class StopWatch(object):

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def reset(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if not self.start_time:
            raise Exception('Stop watch has not been started yet.')
        self.end_time = time.time()
        return self.get_elapsed_time()

    def get_elapsed_time(self):
        if not self.start_time:
            raise Exception('Stop watch has not been started yet.')
        if not self.end_time:
            raise Exception('Stop watch has not ended yet.')
        return self.end_time - self.start_time

    def lap(self):
        if not self.start_time:
            raise Exception('Stop watch has not been started yet.')
        return time.time() - self.start_time
