"""Shared utilities."""

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """Context manager that measures elapsed time in milliseconds."""
    start = time.perf_counter()
    result = {"ms": 0.0}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000
