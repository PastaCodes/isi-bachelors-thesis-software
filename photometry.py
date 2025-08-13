from typing import Final

import numpy as np
import scipy.optimize as spo

from main import MinorPlanetObservation, earth_sun_distance, elongation_from_observation


def phi(phase: float, slope: float) -> float:
    """
    Function Φ(α) as in the HG model, first described by Bowell et al. 1989.
    :param phase: [rad]
    :param slope: Phase slope parameter G [rad]
    """
    half_tan = np.tan(phase / 2)
    return (1 - slope) * np.exp(-3.33 * np.pow(half_tan, 0.63)) + slope * np.exp(-1.87 * np.pow(half_tan, 1.22))


def phi_prime(phase: float, slope: float) -> float:
    """
    Derivative of the above.
    """
    half_tan = np.tan(phase / 2)
    half_sec = 1 / np.cos(phase / 2)
    return -(half_sec * half_sec) * (
            1.05 * (1 - slope) * np.pow(half_tan, -0.37) * np.exp(-3.33 * np.pow(half_tan, 0.63)) +
                1.14 * slope * np.pow(half_tan, 0.22) * np.exp(-1.87 * np.pow(half_tan, 1.22)))


# https://www.minorplanetcenter.net/iau/info/BandConversion.txt
VISUAL_CORRECTION: Final[dict[str, float]] = \
    {' ': -0.8, 'U': -1.3, 'B': -0.8, 'g': -0.35, 'V': 0, 'r': 0.14, 'R': 0.4, 'C': 0.4, 'W': 0.4, 'i': 0.32,
     'z': 0.26, 'I': 0.8, 'J': 1.2, 'w': -0.13, 'y': 0.32, 'L': 0.2, 'H': 1.4, 'K': 1.7, 'Y': 0.7, 'G': 0.28, 'v': 0,
     'c': -0.05, 'o': 0.33, 'u': 2.5}


def visual_magnitude_from_observed(observed_magnitude: float, band: str) -> float:
    return observed_magnitude + VISUAL_CORRECTION[band]


def phase_from_magnitude(obs: MinorPlanetObservation, elongation: float | None = None,
                         es_dist: float | None = None) -> float:
    if elongation is None:
        elongation = elongation_from_observation(obs)
    if es_dist is None:
        es_dist = earth_sun_distance(obs.obstime)

    # print(f'G = {obs.obj.slope}, R = {es_dist}, θ = {elongation}, V = {visual_magnitude}, H = {obs.obj.absolute_magnitude}')

    mag_diff = obs.visual_magnitude - obs.obj.absolute_magnitude
    elong_sin = np.sin(elongation)

    # Phase is governed by the equation f(α)=0, where f is not easily invertible,
    # so we use the Newton-Raphson method to find a root.
    # In general, f may have more than one root, however it is not the case in our observations.
    # In the presence of several roots, one might use a prediction of the phase as an initial guess.

    def f(a: float) -> float:
        sin_a = np.sin(a)
        return (es_dist * es_dist * elong_sin * np.sin(a + elongation) -
                np.pow(10, 0.2 * mag_diff) * sin_a * sin_a * np.sqrt(phi(a, obs.obj.slope)))

    # def f_prime(a: float) -> float:
    #     return (es_dist * es_dist * elong_sin * np.cos(a + elongation) - np.pow(10, 0.2 * mag_diff) *
    #             np.sin(a) * np.cos(a) * np.pow(phi(a, obs.obj.slope), -0.5) * phi_prime(a, obs.obj.slope))

    # return spo.newton(f, phase_guess, f_prime)

    return spo.brentq(f, 0, np.pi)


def distance_from_magnitude(obs: MinorPlanetObservation) -> float:
    elongation = elongation_from_observation(obs)
    es_dist = earth_sun_distance(obs.obstime)
    phase = phase_from_magnitude(obs, elongation, es_dist)
    return es_dist * np.sin(phase + elongation) / np.sin(phase)
