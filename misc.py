import calendar
import datetime as dt
import math


# Rough estimate of the standard deviation based on the number of provided decimals
def sd(decimals: str) -> float:
    precision = len(decimals.replace(' ', ''))
    return 0.5 * 10 ** -precision


class DecimalDayDatetime:
    def __init__(self, year: int, month: int, day_dec: float) -> None:
        self.year = year
        self.month = month
        self.day_dec = day_dec

    def to_datetime(self) -> dt.datetime:
        day_frac, day = math.modf(self.day_dec)
        return dt.datetime(self.year, self.month, int(day)) + dt.timedelta(days=day_frac)

    def to_decimal_year(self) -> float:
        day_frac, day = math.modf(self.day_dec)
        year_day_dec = dt.datetime(self.year, self.month, int(day)).timetuple().tm_yday + day_frac
        return self.year + (year_day_dec - 1) / (366 if calendar.isleap(self.year) else 365)


def find_all_roots(func, interval, step=0.01, rtol=1e-6, atol=1e-8):
    """
    Find all roots of `func` in `interval` by subdivision and Brent's method.

    Parameters:
        func (callable): The function to find roots of.
        interval (tuple): (a, b) - the interval to search.
        step (float): Step size for initial subdivision.
        rtol (float): Relative tolerance for root merging.
        atol (float): Absolute tolerance for root merging.

    Returns:
        list: A list of unique roots sorted in ascending order.
    """
    import numpy as np
    from scipy.optimize import brentq
    a, b = interval
    roots = []

    # Subdivide the interval and search for sign changes
    x = np.arange(a, b + step, step)
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        try:
            root = brentq(func, x1, x2, rtol=rtol)
            roots.append(root)
        except:
            continue

    # Remove duplicate roots (within tolerance)
    unique_roots = []
    for root in sorted(roots):
        if not unique_roots or np.abs(root - unique_roots[-1]) > atol:
            unique_roots.append(root)

    return unique_roots
