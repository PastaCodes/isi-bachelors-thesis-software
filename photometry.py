from typing import Final

import numpy as np
import scipy.optimize as spo
import astropy.units as u
from astropy.units import Quantity

from main import MinorPlanetObservation


# https://www.minorplanetcenter.net/iau/info/BandConversion.txt
VISUAL_CORRECTION: Final[dict[str, float]] = \
    {' ': -0.8, 'U': -1.3, 'B': -0.8, 'g': -0.35, 'V': 0, 'r': 0.14, 'R': 0.4, 'C': 0.4, 'W': 0.4, 'i': 0.32,
     'z': 0.26, 'I': 0.8, 'J': 1.2, 'w': -0.13, 'y': 0.32, 'L': 0.2, 'H': 1.4, 'K': 1.7, 'Y': 0.7, 'G': 0.28, 'v': 0,
     'c': -0.05, 'o': 0.33, 'u': 2.5}


def visual_magnitude_from_observed(observed_magnitude: Quantity, band: str) -> Quantity:
    return observed_magnitude + VISUAL_CORRECTION[band] * u.mag


def phi(phase: float, slope: float) -> Quantity:
    half_tan = np.tan(phase / 2)
    return (1 - slope) * np.exp(-3.33 * np.pow(half_tan, 0.63)) + slope * np.exp(-1.87 * np.pow(half_tan, 1.22))


def phase_from_magnitude(obs: MinorPlanetObservation, visual_magnitude: Quantity, elongation: Quantity,
                         sun_observer_dist: Quantity, tol: float = 2e-12) -> Quantity:
    mag_diff = visual_magnitude.to(u.mag).value - obs.target_body.absolute_magnitude.to(u.mag).value
    th = elongation.to(u.rad).value
    sin_th = np.sin(th)
    rr = sun_observer_dist.to(u.au).value
    gg = obs.target_body.slope.value

    # Phase is governed by the equation f(α)=0, where f is not easily invertible,
    # so we use the Newton-Raphson method to find a root.
    # In general, f may have more than one root, however it is not the case in our observations.
    # In the presence of several roots, one might use a prediction of the phase as an initial guess.

    def f(a: float) -> float:
        sin_a = np.sin(a)
        return (rr * rr * sin_th * np.sin(a + th) -
                np.pow(10, 0.2 * mag_diff) * sin_a * sin_a * np.sqrt(phi(a, gg)))

    return spo.brentq(f, 0, np.pi, xtol=tol) * u.rad


def distance_from_phase(phase: Quantity, elongation: Quantity, sun_observer_dist: Quantity) -> Quantity:
    return sun_observer_dist * np.sin(phase + elongation) / np.sin(phase)


def distance_from_magnitude(obs: MinorPlanetObservation, visual_magnitude: Quantity, elongation: Quantity,
                            sun_observer_dist: Quantity) -> Quantity:
    phase = phase_from_magnitude(obs, visual_magnitude, elongation, sun_observer_dist)
    return distance_from_phase(phase, elongation, sun_observer_dist)
