import datetime as dt
import math

import numpy as np
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time
from astropy.units import Quantity


def safe_acos(a):
    return np.acos(np.clip(a, -1.0, 1.0))


# Rough estimate of the standard deviation based on the number of provided decimals
def sd(decimals: str) -> float:
    precision = len(decimals.replace(' ', ''))
    return 0.5 * 10 ** -precision


def decimal_day_date_to_time(year: int, month: int, day_dec: float, scale: str = 'utc') -> Time:
    day_frac, day = math.modf(day_dec)
    datetime = dt.datetime(year, month, int(day)) + dt.timedelta(days=day_frac)
    return Time(datetime, scale=scale)


def location_from_parallax(lon: Angle, p1: Quantity, p2: Quantity) -> EarthLocation:
    x = p1 * np.cos(lon)
    y = p1 * np.sin(lon)
    z = p2
    return EarthLocation.from_geocentric(x, y, z)
