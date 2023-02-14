import logging
import sys
import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

FORMATTER = logging.Formatter(u'%(asctime)s — %(name)s — %(levelname)s — %(message)s')


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%Y_%m_%d")
    log_dir = Path.home() / "smb_experiments_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"experiments_{date_time}.log"
    Path(log_file).touch(exist_ok=True)
    file_handler = TimedRotatingFileHandler(log_file, when='midnight', encoding='utf-8')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
