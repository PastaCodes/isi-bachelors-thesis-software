import datetime as dt
import math

import numpy as np
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity


def safe_acos(a: float | Quantity) -> float | Quantity:
    return np.acos(np.clip(a, -1.0, 1.0))


def cot(a: float | Quantity) -> float | Quantity:
    if isinstance(a, Quantity):
        return np.tan(np.pi / 2 * u.rad - a)
    else:
        return np.tan(np.pi / 2 - a)


def atan2pos(y: Quantity, x: Quantity) -> Quantity:
    v = np.atan2(y, x)
    return v if v >= 0 else 2 * np.pi * u.rad + v


def law_of_cosines(adj1: float | Quantity, adj2: float | Quantity, opp: float | Quantity) -> float | Quantity:
    return np.acos((adj1 * adj1 + adj2 * adj2 - opp * opp) / (2 * adj1 * adj2))


def closest_angle(choice1: Quantity, choice2: Quantity, reference: Quantity) -> Quantity:
    diff1 = np.abs(choice1 - reference)
    diff1 = np.minimum(diff1, 2 * np.pi * u.rad - diff1)
    diff2 = np.abs(choice2 - reference)
    diff2 = np.minimum(diff2, 2 * np.pi * u.rad - diff2)
    if diff2 < diff1:
        return choice2
    return choice1


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
