"""
Timing utilities for measuring how long operations take (used to power
the statistics panel and general performance logging).
"""
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator

from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timed_block(label: str) -> Iterator[dict]:
    """
    Context manager that measures elapsed time for a block of code.

    Usage:
        with timed_block("embedding") as t:
            ...
        print(t["elapsed"])

    Args:
        label: A human-readable label used in the log line.

    Yields:
        A dict that will be populated with 'elapsed' (seconds) once the
        block finishes.
    """
    start = time.perf_counter()
    result = {"elapsed": None}
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        logger.debug(f"{label} took {result['elapsed']:.3f}s")


def timed(label: str = None):
    """
    Decorator that logs and returns execution time alongside the
    wrapped function's return value is NOT altered — timing is only
    logged, not returned, to keep call sites simple.

    Args:
        label: Optional label; defaults to the function's name.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{name} took {elapsed:.3f}s")
            return result
        return wrapper
    return decorator
