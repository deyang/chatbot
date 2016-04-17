import logging
from logging.handlers import RotatingFileHandler
import os

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
