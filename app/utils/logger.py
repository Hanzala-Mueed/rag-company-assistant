import logging

from app.config.logging_config import setup_logging

setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Return configured logger instance.
    """
    return logging.getLogger(name)