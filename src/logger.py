import os
import sys
import logging

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG


def get_logger(filepath, with_time=True, level=INFO):
    name = os.path.splitext(os.path.basename(filepath))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    format_str = '%(name)s:%(lineno)d - %(levelname)s - %(message)s'
    if with_time:
        format_str = '%(asctime)s - ' + format_str
    handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(handler)
    return logger
