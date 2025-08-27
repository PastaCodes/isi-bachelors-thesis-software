import math

import numpy as np
import scipy as sp

from misc import safe_arccos, cot, closest_angle, arctan2pos, angle_components_sum, wrap_angle


SUN_GRAVITY = 2.959122083E-4  # Standard gravitational parameter μ of the Sun in AU^3/day^2, TDB compatible


def get_mean_motion(a: float) -> float:
    return np.sqrt(SUN_GRAVITY / a ** 3)


def compute_beta(e: float) -> float:
    return e / (1 + np.sqrt(1 - e * e))


def advance_mean_anomaly(before: float, n: float, dt: float) -> float:
    return wrap_angle(before + n * dt)


def eccentric_anomaly_from_distance(r: float, a: float, e: float, mm_hint: float) -> float:
    if mm_hint < np.pi:
        return safe_arccos((1.0 - r / a) / e)
    else:
        return 2.0 * np.pi - safe_arccos((1.0 - r / a) / e)


def distance_from_eccentric_anomaly(ee: float, a: float, e: float) -> float:
    return a * (1 - e * np.cos(ee))


def mean_anomaly_from_eccentric_anomaly(ee: float, e: float) -> float:
    return ee - e * np.sin(ee)


def eccentric_anomaly_from_mean_anomaly(mm: float, e: float, tol: float = 1.48e-08) -> float:
    return sp.optimize.newton(func=(lambda _ee: _ee - e * np.sin(_ee) - mm),
                              fprime=(lambda _ee: 1.0 - e * np.cos(_ee)),
                              fprime2=(lambda _ee: e * np.sin(_ee)),
                              x0=mm, tol=tol)


def true_anomaly_from_eccentric_anomaly(ee: float, e: float) -> float:
    beta = compute_beta(e)
    return ee + 2 * np.atan(beta * np.sin(ee) / (1 - beta * np.cos(ee)))


def orbital_angles_from_position(pos: np.ndarray, v: float, i: float,
                                 om_hint: float) -> tuple[float, float]:
    x, y, z = pos
    om_base = np.atan2(y, x)
    om_offset = np.asin(cot(i) * z / np.sqrt(x * x + y * y))
    om = closest_angle(om_base - om_offset, om_base + np.pi + om_offset, om_hint)
    u = arctan2pos(z, np.sin(i) * (np.cos(om) * x + np.sin(om) * y))
    w = u - v
    return om, w


def position_from_orbital_angle_components(cos_om: float, sin_om: float, cos_w: float, sin_w: float,
                                           cos_v: float, sin_v: float, cos_i: float, sin_i: float,
                                           r: float) -> np.ndarray:
    cos_u, sin_u = angle_components_sum(cos_w, sin_w, cos_v, sin_v)
    return r * np.array([cos_om * cos_u - sin_om * sin_u * cos_i,
                         sin_om * cos_u + cos_om * sin_u * cos_i,
                         sin_u * sin_i])
