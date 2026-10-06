import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    result = {"elapsed_seconds": 0.0}

    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start


def measure_time():
    return time.perf_counter()
