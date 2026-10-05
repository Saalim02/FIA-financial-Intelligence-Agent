import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start


def measure_time():
    return time.perf_counter()
