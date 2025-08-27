from typing import Final

import numpy as np
import scipy as sp


# https://www.minorplanetcenter.net/iau/info/BandConversion.txt
VISUAL_CORRECTION: Final[dict[str, float]] = \
    {' ': -0.8, 'U': -1.3, 'B': -0.8, 'g': -0.35, 'V': 0, 'r': 0.14, 'R': 0.4, 'C': 0.4, 'W': 0.4, 'i': 0.32,
     'z': 0.26, 'I': 0.8, 'J': 1.2, 'w': -0.13, 'y': 0.32, 'L': 0.2, 'H': 1.4, 'K': 1.7, 'Y': 0.7, 'G': 0.28, 'v': 0,
     'c': -0.05, 'o': 0.33, 'u': 2.5}


def visual_magnitude_from_observed(observed_magnitude: float, band: str) -> float:
    return observed_magnitude + VISUAL_CORRECTION[band]


def hg_phi(phase: float, slope: float) -> float:
    half_tan = np.tan(phase / 2)
    return (1 - slope) * np.exp(-3.33 * np.pow(half_tan, 0.63)) + slope * np.exp(-1.87 * np.pow(half_tan, 1.22))


def visual_magnitude_from_absolute(hh: float, d: float, delta: float, phi: float, gg: float) -> float:
    return hh + 5.0 * np.log10(d * delta) - 2.5 * np.log10(hg_phi(phi, gg))


def solve_distance_and_phase(vv: float, rr: float, th: float, hh: float, gg: float,
                             tol: float = 2e-12) -> tuple[float, float]:
    mag_diff_term = np.power(10.0, 0.2 * (vv - hh))
    sin_th = np.sin(th)
    delta_sin_phi = np.nan
    sin_phi = np.nan

    def phase_equation(_phi: float) -> float:
        nonlocal delta_sin_phi, sin_phi
        d_sin_phi = rr * sin_th  # By the law of sines
        delta_sin_phi = rr * np.sin(th + _phi)  # By the law of sines
        sin_phi = np.sin(_phi)
        return d_sin_phi * delta_sin_phi - sin_phi * sin_phi * np.sqrt(hg_phi(_phi, gg)) * mag_diff_term

    phi = sp.optimize.brentq(phase_equation, 0.0, np.pi, xtol=tol)
    delta = delta_sin_phi / sin_phi
    return delta, phi
