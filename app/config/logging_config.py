import logging

from app.config.settings import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)

APP_LOG_FILE = LOG_DIR / "app.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"


def setup_logging():
    """
    Configure application logging.
    """

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Application log file
    file_handler = logging.FileHandler(
        APP_LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Error log file
    error_handler = logging.FileHandler(
        ERROR_LOG_FILE,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)