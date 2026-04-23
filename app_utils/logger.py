"""
Logger utility — setup logging cho project.
"""

import logging
import os


def get_logger(name="iad", log_file=None):
    """
    Tạo logger với console output và optional file output.

    Args:
        name: tên logger
        log_file: (optional) đường dẫn file log

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Tránh tạo duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
