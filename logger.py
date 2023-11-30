import logging
from typing import Optional


def get_clean_logger(logger_name: str = 'no_spam', log_filename: Optional[str] = None):
    """Gets a logger with no BS"""
    if log_filename is not None:
        logging.basicConfig(filename=log_filename, filemode='w')

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger