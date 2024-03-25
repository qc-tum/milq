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


def time_conversion(
    time: float, unit: str, target_unit: str = "us", dt: float | None = None
) -> float:
    """Converts a time from one unit to another.

    Args:
        time (float): The time to convert.
        unit (str): The unit of the time.
        target_unit (str, optional): The target unit. Defaults to "us".
        dt (float | None, optional): The duration in seconds of the device-dependent
        time. Must be set if unit is in dt but target isn't. Defaults to None.

    Returns:
        float: The time in the target unit.
    """
    if unit == target_unit:
        return time

    units = ["s", "ms", "us", "ns", "ps"]

    # target_unit must be a SI unit
    assert target_unit in units

    # Convert dt (device-dependent time) to SI unit
    if unit == "dt":
        assert dt is not None
        time *= dt
        unit = "s"

    target_shift = units.index(target_unit)
    current_shift = units.index(unit)
    required_shift = 3 * (target_shift - current_shift)
    return time * 10**required_shift
