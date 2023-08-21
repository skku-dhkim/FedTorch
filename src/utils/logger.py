"""
@ File name: logger.py
@ Version: 1.1.0
@ Last update: 2021.Aug.25
@ Author: DH.KIM
@ Copy rights: SKKU Applied AI and Computer Vision Lab
@ Description: Colorful logger
"""
import coloredlogs
import logging
import os

from conf.logger_config import field_styles, level_styles, LOGGER_DICT
from typing import Optional, Tuple

LEVEL_DICT = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'CRITICAL': logging.CRITICAL
}


def mkdir_p(path):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        raise


class FedTorchFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        mkdir_p(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)


def get_logger(name: str,
               log_path: Optional[str] = None,
               level: str = "DEBUG",
               log_type: str = "stream") -> Tuple[logging.Logger, str]:

    if level.upper() not in LEVEL_DICT.keys():
        raise ValueError("Invalid Logger level. You got: '{}'".format(level))

    # INFO: Check Logger if exists.
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name), name

    if log_type == "file":
        if log_path is not None:
            name = "fedtorch.file.{}".format(name)
            logger = logging.getLogger(name)

            log_handler = FedTorchFileHandler(filename=log_path)
            log_handler.setLevel(LEVEL_DICT[level])

            formatter = logging.Formatter('[%(asctime)s,%(msecs)03d] %(name)s[%(process)d] [%(levelname)s] %(message)s')
            log_handler.setFormatter(formatter)

            logger.addHandler(log_handler)
            logger.setLevel(LEVEL_DICT[level])
            logger.propagate = False
            return logger, name
        else:
            raise ValueError("Log path is required for log type \'file\'.")
    else:
        name = "fedtorch.stream.{}".format(name)

        logger = logging.getLogger(name)

        log_handler = logging.StreamHandler()
        log_handler.setLevel(LEVEL_DICT[level.upper()])

        formatter = coloredlogs.ColoredFormatter(
            '[%(asctime)s,%(msecs)03d] %(name)s[%(process)d] [%(levelname)s] %(message)s',
            field_styles=field_styles,
            level_styles=level_styles)
        log_handler.setFormatter(formatter)

        logger.addHandler(log_handler)
        logger.setLevel(LEVEL_DICT[level.upper()])
        logger.propagate = False
        return logger, name


def write_experiment_summary(title: str, context: dict) -> None:
    logger, _ = get_logger(LOGGER_DICT['summary'], log_type='file')
    logger.info("*" * 15 + " {} ".format(title) + "*" * 15)
    for key, value in context.items():
        logger.info("{}: {}".format(key, value))
