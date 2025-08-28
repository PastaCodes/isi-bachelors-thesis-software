import datetime as dt
import math
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time


PROJECT_ROOT = str(Path(sys.prefix).parent) + '/'


def wrap_angle(a: float) -> float:
    return a - 2 * np.pi * np.floor(a / (2 * np.pi))


def safe_arccos(a: float) -> float:
    return np.arccos(np.clip(a, -1.0, 1.0))


def safe_arcsin(a: float) -> float:
    return np.arcsin(np.clip(a, -1.0, 1.0))


def cot(a: float) -> float:
    return np.tan(np.pi / 2 - a)


def arctan2pos(y: float, x: float) -> float:
    return wrap_angle(np.arctan2(y, x))


def angle_components(a: float) -> tuple[float, float]:
    return np.cos(a), np.sin(a)


def angle_from_components(cos: float, sin: float) -> float:
    return arctan2pos(sin, cos)


def angle_components_sum(cos_a: float, sin_a: float, cos_b: float, sin_b: float) -> tuple[float, float]:
    return cos_a * cos_b - sin_a * sin_b, sin_a * cos_b + cos_a * sin_b


def law_of_cosines(adj1: float, adj2: float, opp: float) -> float:
    return safe_arccos((adj1 * adj1 + adj2 * adj2 - opp * opp) / (2 * adj1 * adj2))


def angle_dist(a1: float, a2: float) -> float:
    a1 = wrap_angle(a1)
    a2 = wrap_angle(a2)
    diff = np.abs(a1 - a2)
    return np.minimum(diff, 2 * np.pi - diff)


def closest_angle(a1: float, a2: float, ref: float) -> float:
    return a2 if angle_dist(a2, ref) < angle_dist(a1, ref) else a1


def norm(vec: np.ndarray) -> float:
    # noinspection PyTypeChecker
    return np.linalg.norm(vec)


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / norm(vec)


def unit_vector_separation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return safe_arccos(vec1.dot(vec2))


def decimal_day_date_to_time(year: int, month: int, day_dec: float, location: EarthLocation,
                             scale: str = 'utc') -> Time:
    day_frac, day = math.modf(day_dec)
    datetime = dt.datetime(year, month, int(day)) + dt.timedelta(days=day_frac)
    return Time(datetime, scale=scale, location=location)


def location_from_parallax(lon_deg: float, p1: float, p2: float) -> EarthLocation:
    lon = np.radians(lon_deg)
    x = p1 * np.cos(lon)
    y = p1 * np.sin(lon)
    z = p2
    return EarthLocation.from_geocentric(x, y, z, unit=u.earthRad)
