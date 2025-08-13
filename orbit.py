import math

import numpy as np
import scipy.optimize as spo


SUN_GRAVITY = 39.4769264145  # Standard gravitational parameter μ of the Sun in AU^3/y^2


def get_mean_motion(semi_major_axis: float) -> float:
    return math.sqrt(SUN_GRAVITY / semi_major_axis ** 3)


def advance_mean_anomaly(before: float, mean_motion: float, delta_time: float) -> float:
    return math.fmod(before + mean_motion * delta_time, 2 * math.pi)


def eccentric_anomaly_from_distance(distance: np.number, semi_major_axis: float, eccentricity: float,
                                    mean_anomaly_hint: float) -> float:
    if mean_anomaly_hint < np.pi:
        return math.acos((1 - distance / semi_major_axis) / eccentricity)
    else:
        return 2 * math.pi - math.acos((1 - distance / semi_major_axis) / eccentricity)


def distance_from_eccentric_anomaly(eccentric_anomaly: float, semi_major_axis: float, eccentricity: float) -> float:
    return semi_major_axis * (1 - eccentricity * math.cos(eccentric_anomaly))


def mean_anomaly_from_eccentric_anomaly(eccentric_anomaly: float, eccentricity: float) -> float:
    return eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)


def eccentric_anomaly_from_mean_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    def f(e: float) -> float:
        return e - eccentricity * math.sin(e) - mean_anomaly
    def f_prime(e: float) -> float:
        return 1 - eccentricity * math.cos(e)
    return spo.newton(f, mean_anomaly, f_prime)


def true_anomaly_from_eccentric_anomaly(eccentric_anomaly: float, beta: float) -> float:
    return eccentric_anomaly + 2 * math.atan(beta * math.sin(eccentric_anomaly) /
                                             (1 - beta * math.cos(eccentric_anomaly)))


def z_angles(position: np.ndarray, true_anomaly: float, cot_i: float) -> tuple[float, float]:
    x, y, z = position
    z_angle = math.atan2(math.sqrt(x ** 2 + y ** 2 + (1 - cot_i ** 2) * z ** 2), -z * cot_i)
    z_angle_offset = z_angle - true_anomaly

    # TODO in general, which solution??
    # orbit_plane_angle = math.atan2(y, x) + math.acos(-z * cot_i / math.sqrt(x ** 2 + y ** 2))
    orbit_plane_angle = math.atan2(y, x) - math.acos(-z * cot_i / math.sqrt(x ** 2 + y ** 2))

    return z_angle_offset, orbit_plane_angle


def position_from_orbit_angles(true_anomaly: float, z_angle_offset: float, orbit_plane_angle: float,
                               inclination: float, distance: float) -> np.ndarray:
    z_angle = true_anomaly + z_angle_offset
    from scipy.spatial.transform import Rotation
    position = np.array([distance, 0.0, 0.0])
    return Rotation.from_euler('zyz', [z_angle, inclination, orbit_plane_angle]).apply(position)
