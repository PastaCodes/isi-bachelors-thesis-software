import numpy as np
import scipy as sp
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy.spatial.transform import Rotation

from misc import (wrap_angle_f, arctan2pos_f, safe_arcsin_f, law_of_cosines_f, unit_vector_separation_f,
                  angle_components_f, safe_arccos_f)
from photometry import phi_f


"""
Ideal model

Simulates elliptical orbits of Bennu, the Sun, and an observer around the Solar System barycenter, assuming their
orbital elements to be constant.
In particular, values for the semi major axis, eccentricity, inclination, longitude of the ascending node, argument of
periapsis, mean motion, and time of periapsis are taken to be accurate, for all three subjects, at the time
2000-Jan-01 00:00:00.0000 Julian Date, Barycentric Dynamical Time.
Observations are also simplified to ignore light dynamics, so that the apparent right ascension and declination are
equal to the respective geometric values.
In a more accurate model, the observer would be positioned on the Earth's surface, but in this case it is aligned with
the Earth-Moon barycenter, which should nevertheless provide sufficient variation in the data.
"""


TGT_A = 1.129706435415488E+00                         # Bennu orbit semi major axis
TGT_E = 2.093586393061045E-01                         # Bennu orbit eccentricity
TGT_I = np.radians(2.948665502983759E+01)             # Bennu orbit inclination
TGT_O = np.radians(4.893785334998477E-01)             # Bennu orbit longitude of the ascending node
TGT_W = np.radians(6.587502788515098E+01)             # Bennu orbit argument of periapsis
TGT_N = np.radians(8.213851581515027E-01)             # Bennu orbit mean motion
TGT_B = TGT_E / (1.0 + np.sqrt(1.0 - TGT_E * TGT_E))  # Bennu orbit beta value
TGT_T = 2.451500519003912E+06                         # Bennu orbit time of periapsis JD TDB
TGT_H = 20.21                                         # Bennu absolute magnitude
TGT_G = -0.031                                        # Bennu slope parameter

OBS_A = 9.950701204601046E-01                         # Observer orbit semi major axis
OBS_E = 1.473537140424812E-02                         # Observer orbit eccentricity
OBS_I = np.radians(2.345112700862180E+01)             # Observer orbit inclination
OBS_O = np.radians(4.629567569877889E-03)             # Observer orbit longitude of the ascending node
OBS_W = np.radians(7.459613421556121E+01)             # Observer orbit argument of periapsis
OBS_N = np.radians(9.936026676711932E-01)             # Observer orbit mean motion
OBS_B = OBS_E / (1.0 + np.sqrt(1.0 - OBS_E * OBS_E))  # Observer orbit beta value
OBS_T = 2.451519356516061E+06                         # Observer orbit time of periapsis JD TDB

SUN_A = 6.981597616379102E-03                         # Sun orbit semi major axis
SUN_E = 2.694756164323808E-01                         # Sun orbit eccentricity
SUN_I = np.radians(2.311817382515920E+01)             # Sun orbit inclination
SUN_O = np.radians(3.852448313103996E+00)             # Sun orbit longitude of the ascending node
SUN_W = np.radians(3.231307573061241E+02)             # Sun orbit argument of periapsis
SUN_N = np.radians(8.293546641817494E-02)             # Sun orbit mean motion
SUN_B = SUN_E / (1.0 + np.sqrt(1.0 - SUN_E * SUN_E))  # Sun orbit beta value
SUN_T = 2.452714888563351E+06                         # Sun orbit time of periapsis JD TDB

DIR_PREC = 1e15  # Precision of the observation direction (concentration parameter of the Fisher distribution)
MAG_SIGMA = 0.01   # Standard deviation of the visual magnitude


def real_position(t: float, a: float, e: float, i: float, om: float, w: float, n: float, b: float,
                  tp: float) -> np.ndarray:
    mean_anom = wrap_angle_f(n * (t - tp))

    def f(ee: float) -> float:
        return ee - e * np.sin(ee) - mean_anom

    def f_prime(ee: float) -> float:
        return 1.0 - e * np.cos(ee)

    ecc_anom = sp.optimize.newton(f, mean_anom, f_prime)
    true_anom = ecc_anom + 2.0 * np.atan(b * np.sin(ecc_anom) / (1.0 - b * np.cos(ecc_anom)))
    dist = a * (1.0 - e * np.cos(ecc_anom))
    return Rotation.from_euler('zxz', [true_anom + w, i, om]).apply(np.array([dist, 0.0, 0.0]))


def gen_times(n: int, min_dist: float, max_dist: float, start: float, rng: np.random.Generator) -> list[float]:
    acc = start
    times = []
    for i in range(n):
        times.append(acc)
        acc += rng.uniform(min_dist, max_dist)
    return times


def observe(tgt_pos: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray, h: float, g: float,
            direction_precision: float, mag_sigma: float, rng: np.random.Generator) -> np.ndarray:
    tgt_obs_pos = tgt_pos - obs_pos
    tgt_obs_dist = np.linalg.norm(tgt_obs_pos)

    tgt_obs_ray = tgt_obs_pos / tgt_obs_dist
    # Apply Fisher noise to the observation direction
    tgt_obs_ray = sp.stats.vonmises_fisher(mu=tgt_obs_ray, kappa=direction_precision, seed=rng).rvs(1)[0]
    ra = arctan2pos_f(tgt_obs_ray[1], tgt_obs_ray[0])
    dec = safe_arcsin_f(tgt_obs_ray[2])

    tgt_sun_dist = np.linalg.norm(tgt_pos - sun_pos)
    obs_sun_dist = np.linalg.norm(obs_pos - sun_pos)

    phase = law_of_cosines_f(tgt_obs_dist, tgt_sun_dist, obs_sun_dist)
    phi_correction = 2.5 * np.log10(phi_f(phase, g))
    reduced_mag = h - phi_correction
    v = reduced_mag + 5.0 * np.log10(tgt_sun_dist * tgt_obs_dist)
    v = rng.normal(loc=v, scale=mag_sigma)  # Apply Gaussian noise to the visual magnitude
    return np.array([ra, dec, v])


def prepare(obs: np.ndarray) -> np.ndarray:
    ra, dec, v = obs
    cos_ra = np.cos(ra)
    sin_ra = np.sin(ra)
    cos_dec = np.cos(dec)
    sin_dec = np.sin(dec)
    return np.array([cos_ra, sin_ra, cos_dec, sin_dec, v])


def reverse_transform(obs_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
                      h: float, g: float) -> np.ndarray:
    """
    [*α, *δ, V] --> [x, y, z]
    """
    cos_ra, sin_ra, cos_dec, sin_dec, v = obs_vec
    tgt_obs_unit_vec = np.array([cos_ra * cos_dec, sin_ra * cos_dec, sin_dec])
    sun_obs_pos = sun_pos - obs_pos
    sun_obs_dist = np.linalg.norm(sun_obs_pos)
    sun_obs_unit_vec = sun_obs_pos / sun_obs_dist
    elong = unit_vector_separation_f(tgt_obs_unit_vec, sun_obs_unit_vec)
    mag_diff = v - h
    elong_sin = np.sin(elong)

    def f(a: float) -> float:
        sin_a = np.sin(a)
        return (sun_obs_dist * sun_obs_dist * elong_sin * np.sin(a + elong) -
                np.pow(10.0, 0.2 * mag_diff) * sin_a * sin_a * np.sqrt(phi_f(a, g)))

    phase = sp.optimize.brentq(f, 0.0, np.pi)
    tgt_obs_dist = sun_obs_dist * np.sin(phase + elong) / np.sin(phase)
    tgt_obs_pos = tgt_obs_dist * tgt_obs_unit_vec
    tgt_pos = tgt_obs_pos + obs_pos
    return tgt_pos


def initial_state(obs_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
                  a: float, e: float, i: float, o: float, w: float, n: float, h: float, g: float,
                  m_hint: float) -> np.ndarray:
    tgt_pos = reverse_transform(obs_vec, obs_pos, sun_pos, h, g)
    r = np.linalg.norm(tgt_pos)
    if m_hint < np.pi:
        ee = safe_arccos_f((1.0 - r / a) / e)
    else:
        ee = 2.0 * np.pi - safe_arccos_f((1.0 - r / a) / e)
    m = ee - e * np.sin(ee)
    cos_m, sin_m = angle_components_f(m)
    cos_i, sin_i = angle_components_f(i)
    cos_o, sin_o = angle_components_f(o)
    cos_w, sin_w = angle_components_f(w)
    return np.array([cos_m, sin_m, a, e, cos_i, sin_i, cos_o, sin_o, cos_w, sin_w, n, h, g])


def finalize(state_vec: np.ndarray) -> np.ndarray:
    """
    [*M, a, e, *i, *Ω, *ω, n, H, G] --> [x, y, z]
    """
    cos_m, sin_m, a, e, cos_i, sin_i, cos_o, sin_o, cos_w, sin_w, n, h, g = state_vec
    m = arctan2pos_f(sin_m, cos_m)
    ee = sp.optimize.newton(func=(lambda _ee: _ee - e * np.sin(_ee) - m),
                            fprime=(lambda _ee: 1.0 - e * np.cos(_ee)),
                            x0=m)
    beta = e / (1.0 + np.sqrt(1.0 - e * e))
    true_anom = ee + 2.0 * np.atan(beta * np.sin(ee) / (1.0 - beta * np.cos(ee)))
    cos_v, sin_v = angle_components_f(true_anom)
    cos_u = cos_v * cos_w - sin_v * sin_w
    sin_u = sin_v * cos_w + cos_v * sin_w
    r = a * (1.0 - e * np.cos(ee))
    tgt_pos = r * np.array([cos_o * cos_u - sin_o * sin_u * cos_i,
                            sin_o * cos_u + cos_o * sin_u * cos_i,
                            sin_u * sin_i])
    return tgt_pos


def measure(state_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray) -> np.ndarray:
    """
    [*M, a, e, *i, *Ω, *ω, n, H, G]_k --> [*α, *δ, V]_k
    """
    tgt_pos = finalize(state_vec)
    h, g = state_vec[-2:]
    tgt_obs_pos = tgt_pos - obs_pos
    tgt_obs_dist = np.linalg.norm(tgt_obs_pos)
    tgt_sun_dist = np.linalg.norm(tgt_pos - sun_pos)
    obs_sun_dist = np.linalg.norm(obs_pos - sun_pos)
    sin_dec = tgt_obs_pos[2] / tgt_obs_dist
    cos_dec = np.sqrt(1.0 - sin_dec * sin_dec)
    cos_ra = tgt_obs_pos[0] / (tgt_obs_dist * cos_dec)
    sin_ra = tgt_obs_pos[1] / (tgt_obs_dist * cos_dec)
    phase = law_of_cosines_f(tgt_obs_dist, tgt_sun_dist, obs_sun_dist)
    phi_correction = 2.5 * np.log10(phi_f(phase, g))
    reduced_mag = h - phi_correction
    v = reduced_mag + 5.0 * np.log10(tgt_sun_dist * tgt_obs_dist)
    return np.array([cos_ra, sin_ra, cos_dec, sin_dec, v])


def propagate(before_vec: np.ndarray, dt: float) -> np.ndarray:
    """
    [*M, a, e, *i, *Ω, *ω, n, H, G]_k-1 --> [*M, a, e, *i, *Ω, *ω, n, H, G]_k
    """
    cos_m_before, sin_m_before, a, e, cos_i, sin_i, cos_o, sin_o, cos_w, sin_w, n, h, g = before_vec
    m_before = arctan2pos_f(sin_m_before, cos_m_before)
    m_after = wrap_angle_f(m_before + n * dt)
    ee_after = sp.optimize.newton(func=(lambda _ee: _ee - e * np.sin(_ee) - m_after),
                                  fprime=(lambda _ee: 1.0 - e * np.cos(_ee)),
                                  x0=m_after)
    cos_m_after, sin_m_after = angle_components_f(m_after)
    return np.array([cos_m_after, sin_m_after, a, e, cos_i, sin_i, cos_o, sin_o, cos_w, sin_w, n, h, g])


def main() -> None:
    rng = np.random.default_rng(0)
    times = gen_times(n=150, min_dist=0.01, max_dist=10.0, start=2451544.5, rng=rng)
    tgt_pos = [real_position(t, TGT_A, TGT_E, TGT_I, TGT_O, TGT_W, TGT_N, TGT_B, TGT_T) for t in times]
    obs_pos = [real_position(t, OBS_A, OBS_E, OBS_I, OBS_O, OBS_W, OBS_N, OBS_B, OBS_T) for t in times]
    sun_pos = [real_position(t, SUN_A, SUN_E, SUN_I, SUN_O, SUN_W, SUN_N, SUN_B, SUN_T) for t in times]

    a = TGT_A
    e = TGT_E
    i = TGT_I
    o = TGT_O
    w = TGT_W
    n = TGT_N
    h = TGT_H
    g = TGT_G

    k = DIR_PREC
    v_sig = MAG_SIGMA

    noisy = [observe(t, o, s, TGT_H, TGT_G, DIR_PREC, MAG_SIGMA, rng)
             for t, o, s in zip(tgt_pos, obs_pos, sun_pos)]
    prepared = [prepare(obs) for obs in noisy]

    naive = [reverse_transform(obs, o, s, h, g) for obs, o, s in zip(prepared, obs_pos, sun_pos)]

    points = MerweScaledSigmaPoints(13, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=13, dim_z=5, dt=None, hx=measure, fx=propagate, points=points)

    x0 = initial_state(prepared[0], obs_pos[0], sun_pos[0], a, e, i, o, w, n, h, g, m_hint=0.0)
    pp0 = np.diag([0.5, 0.5, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    prev_t = times[0]
    ukf.x = x0
    ukf.P = pp0

    estimates = [finalize(x0)]
    for t, obs, o, s in zip(times[1:], prepared[1:], obs_pos[1:], sun_pos[1:]):
        dt = t - prev_t
        ukf.Q = 0.0000001 * np.eye(13)
        cos_ra, sin_ra, cos_dec, sin_dec, v = obs
        ukf.R = np.block([[sin_ra * sin_ra / (k * cos_dec * cos_dec),
                           -sin_ra * cos_ra / (k * cos_dec * cos_dec),
                           0, 0, 0],
                          [-sin_ra * cos_ra / (k * cos_dec * cos_dec),
                           cos_ra * cos_ra / (k * cos_dec * cos_dec),
                           0, 0, 0],
                          [0, 0,
                           sin_dec * sin_dec / k,
                           -sin_dec * cos_dec / k,
                           0],
                          [0, 0,
                           -sin_dec * cos_dec / k,
                           cos_dec * cos_dec / k,
                           0],
                          [0, 0, 0, 0, v_sig]])
        ukf.predict(dt=dt)
        ukf.update(z=obs, obs_pos=o, sun_pos=s)
        estimates.append(finalize(ukf.x))
        prev_t = t

    import matplotlib.pyplot as plt
    plt.scatter(times, [np.linalg.norm(n - truth) for n, truth in zip(naive, tgt_pos)], marker='+', c='r')
    plt.scatter(times, [np.linalg.norm(est - truth) for est, truth in zip(estimates, tgt_pos)], marker='+', c='k')
    plt.show()


if __name__ == '__main__':
    main()
