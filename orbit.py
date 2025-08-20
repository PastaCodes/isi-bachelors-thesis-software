import math

import numpy as np
import scipy.optimize as spo
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from astropy.units import Quantity

from misc import safe_arccos, cot, closest_angle, arctan2pos


SUN_GRAVITY = 39.4769264145 * u.au ** 3 / u.year ** 2  # Standard gravitational parameter μ of the Sun


def get_mean_motion(a: Quantity) -> Quantity:
    return np.sqrt(SUN_GRAVITY / a ** 3) * u.rad


def compute_beta(e: Quantity) -> Quantity:
    return e / (1 + np.sqrt(1 - e * e))


def advance_mean_anomaly(before: Quantity, mean_motion: Quantity, delta_time: Quantity) -> Quantity:
    return math.fmod((before + mean_motion * delta_time).to(u.rad).value, 2 * np.pi) * before.unit


def eccentric_anomaly_from_distance(distance: Quantity, semi_major_axis: Quantity, eccentricity: Quantity,
                                    mean_anomaly_hint: Quantity) -> Quantity:
    if mean_anomaly_hint < np.pi * u.rad:
        return safe_arccos((1 - distance / semi_major_axis) / eccentricity)
    else:
        return 2 * np.pi * u.rad - safe_arccos((1 - distance / semi_major_axis) / eccentricity)


def distance_from_eccentric_anomaly(ee: Quantity, a: Quantity, e: Quantity) -> Quantity:
    return a * (1 - e * np.cos(ee))


def mean_anomaly_from_eccentric_anomaly(eccentric_anomaly: Quantity, eccentricity: Quantity) -> Quantity:
    return eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) * u.rad


def eccentric_anomaly_from_mean_anomaly(mean_anomaly: Quantity, eccentricity: Quantity,
                                        tol: float = 1.48e-08) -> Quantity:
    e = eccentricity.value
    mm = mean_anomaly.to(u.rad).value
    def f(ee: float) -> float:
        return ee - e * np.sin(ee) - mm
    def f_prime(ee: float) -> float:
        return 1 - e * np.cos(ee)
    return spo.newton(f, mm, f_prime, tol=tol) * u.rad


def true_anomaly_from_eccentric_anomaly(ee: Quantity, beta: Quantity) -> Quantity:
    return ee + 2 * np.atan(beta * np.sin(ee) / (1 - beta * np.cos(ee)))


def orbital_angles_from_position(position: CartesianRepresentation, true_anomaly: Quantity, incl: Quantity,
                                 asc_long_hint: Quantity) -> tuple[Quantity, Quantity]:
    x, y, z = position.get_xyz()
    asc_long_base = np.atan2(y, x)
    asc_long_offset = np.asin(cot(incl) * z / np.sqrt(x * x + y * y))
    asc_long = closest_angle(asc_long_base - asc_long_offset,
                             asc_long_base + np.pi * u.rad + asc_long_offset,
                             asc_long_hint)
    if asc_long < 0:
        asc_long = 0 * u.rad  # Don't wrap around! The real value must be non-negative.
    lat_arg = arctan2pos(z, np.sin(incl) * (np.cos(asc_long) * x + np.sin(asc_long) * y))
    peri_arg = lat_arg - true_anomaly
    return asc_long, peri_arg


def position_from_orbital_angles(true_anomaly: Quantity, peri_arg: Quantity, asc_long: Quantity,
                                 inclination: Quantity, distance: Quantity) -> CartesianRepresentation:
    lat_arg = true_anomaly + peri_arg
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_euler('zxz', [asc_long.to(u.rad).value,
                                            inclination.to(u.rad).value,
                                            lat_arg.to(u.rad).value])
    position = distance * rotations.apply(np.array([1, 0, 0]))
    return CartesianRepresentation(position)
