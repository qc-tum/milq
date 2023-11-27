"""Helper functions and classes."""
from time import perf_counter


class Timer:
    """A performance timer that can be used as a context manager."""

    def __init__(self):
        self._start = 0
        self._end = 0
        self.elapsed = 0

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, typ, value, traceback):
        self._end = perf_counter()
        self.elapsed = self._end - self._start
