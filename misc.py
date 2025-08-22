import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity


PROJECT_ROOT = str(Path(sys.prefix).parent) + '/'


def wrap_angle(a: Quantity) -> Quantity:
    return wrap_angle_f(a.to_value(u.rad)) * u.rad


def wrap_angle_f(a: float) -> float:
    return a - 2 * np.pi * np.floor(a / (2 * np.pi))


def safe_arccos(a: Quantity) -> Quantity:
    assert a.unit == u.dimensionless_unscaled
    return safe_arccos_f(a.value) * u.rad


def safe_arccos_f(a: float) -> float:
    return np.arccos(np.clip(a, -1.0, 1.0))


def safe_arcsin(a: Quantity) -> Quantity:
    assert a.unit == u.dimensionless_unscaled
    return safe_arcsin_f(a.value) * u.rad


def safe_arcsin_f(a: float) -> float:
    return np.arcsin(np.clip(a, -1.0, 1.0))


def cot(a: Quantity) -> Quantity:
    assert a.unit.is_equivalent(u.rad)
    return cot_f(a.to_value(u.rad)) * u.dimensionless_unscaled


def cot_f(a: float) -> float:
    return np.tan(np.pi / 2 - a)


def arctan2pos(y: Quantity, x: Quantity) -> Quantity:
    assert x.unit.is_equivalent(y.unit)
    return arctan2pos_f(y.value, x.to_value(y.unit)) * u.rad


def arctan2pos_f(y: float, x: float) -> float:
    return wrap_angle_f(np.arctan2(y, x))


def angle_components(a: Quantity) -> Quantity:
    return angle_components_f(a.to_value(u.rad)) * u.rad


def angle_components_f(a: float) -> tuple[float, float]:
    return np.cos(a), np.sin(a)


def law_of_cosines(adj1: Quantity, adj2: Quantity, opp: Quantity) -> Quantity:
    assert adj2.unit.is_equivalent(adj1.unit) and opp.unit.is_equivalent(adj1.unit)
    return law_of_cosines_f(adj1.value, adj2.to_value(adj1.unit), opp.to_value(adj1.unit)) * u.rad


def law_of_cosines_f(adj1: float, adj2: float, opp: float) -> float:
    return safe_arccos_f((adj1 * adj1 + adj2 * adj2 - opp * opp) / (2 * adj1 * adj2))


def angle_dist(a1: Quantity, a2: Quantity) -> Quantity:
    return angle_dist_f(a1.to_value(u.rad), a2.to_value(u.rad)) * u.rad


def angle_dist_f(a1: float, a2: float) -> float:
    a1 = wrap_angle_f(a1)
    a2 = wrap_angle_f(a2)
    diff = np.abs(a1 - a2)
    return np.minimum(diff, 2 * np.pi - diff)


def closest_angle(a1: Quantity, a2: Quantity, ref: Quantity) -> Quantity:
    return closest_angle_f(a1.to_value(u.rad), a2.to_value(u.rad), ref.to_value(u.rad)) * u.rad


def closest_angle_f(a1: float, a2: float, ref: float) -> float:
    return a2 if angle_dist_f(a2, ref) < angle_dist_f(a1, ref) else a1


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def unit_vector_separation_f(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return safe_arccos_f(vec1.dot(vec2))


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
