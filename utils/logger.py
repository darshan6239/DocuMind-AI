"""
Centralized logging configuration for DocuMind AI.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_LEVEL

_LOGGERS = {}

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "documind") -> logging.Logger:
    """
    Get (or create) a configured logger with both console and rotating
    file handlers.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            LOG_DIR / "documind.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
