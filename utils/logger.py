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

from conf.logger_config import field_styles, level_styles

level_dict = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'CRITICAL': logging.CRITICAL
}


def get_stream_logger(name, level: str = "DEBUG"):
    if level.upper() not in level_dict.keys():
        raise ValueError("Invalid Logger level. You got: '{}'".format(level))
    name = "ST_{}".format(name)
    logger = logging.getLogger(name)

    s_logger = logging.StreamHandler()
    s_logger.setLevel(level_dict[level])

    formatter = coloredlogs.ColoredFormatter('%(asctime)s,%(msecs)03d %(name)s[%(process)d] %(levelname)s %(message)s',
                                             field_styles=field_styles,
                                             level_styles=level_styles)
    s_logger.setFormatter(formatter)

    logger.addHandler(s_logger)
    logger.setLevel(level_dict[level])
    logger.propagate = False

    return logger


def get_file_logger(name, log_path, level: str = "DEBUG"):
    if level.upper() not in level_dict.keys():
        raise ValueError("Invalid Logger level. You got: '{}'".format(level))
    logger = logging.getLogger(name)
    f_logger = logging.FileHandler(filename=log_path)
    f_logger.setLevel(level_dict[level])

    formatter = logging.Formatter('%(asctime)s,%(msecs)03d %(name)s[%(process)d] %(levelname)s %(message)s')
    f_logger.setFormatter(formatter)

    logger.addHandler(f_logger)
    logger.setLevel(level_dict[level])
    logger.propagate = False

    return logger
