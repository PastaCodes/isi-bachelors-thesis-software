import numpy as np
import scipy as sp
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from misc import wrap_angle, angle_components, norm
from model import reverse_transform, measure, propagate, initial_state, finalize_transform, \
    state_autocovariance_matrix, measurement_autocovariance_matrix
from orbit import eccentric_anomaly_from_mean_anomaly, true_anomaly_from_eccentric_anomaly, \
    distance_from_eccentric_anomaly, position_from_orbital_angles
from photometry import visual_magnitude_from_absolute
from sky import direction_to_ra_dec, phase_from_distances

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


TGT_A = 1.129706435415488E+00               # Bennu orbit semi major axis
TGT_E = 2.093586393061045E-01               # Bennu orbit eccentricity
TGT_I = np.radians(2.948665502983759E+01)   # Bennu orbit inclination
TGT_OM = np.radians(4.893785334998477E-01)  # Bennu orbit longitude of the ascending node
TGT_W = np.radians(6.587502788515098E+01)   # Bennu orbit argument of periapsis
TGT_N = np.radians(8.213851581515027E-01)   # Bennu orbit mean motion
TGT_TP = 2.451500519003912E+06              # Bennu orbit time of periapsis JD TDB
TGT_HH = 20.21                              # Bennu absolute magnitude
TGT_GG = -0.031                             # Bennu slope parameter

OBS_A = 9.950701204601046E-01               # Observer orbit semi major axis
OBS_E = 1.473537140424812E-02               # Observer orbit eccentricity
OBS_I = np.radians(2.345112700862180E+01)   # Observer orbit inclination
OBS_OM = np.radians(4.629567569877889E-03)  # Observer orbit longitude of the ascending node
OBS_W = np.radians(7.459613421556121E+01)   # Observer orbit argument of periapsis
OBS_N = np.radians(9.936026676711932E-01)   # Observer orbit mean motion
OBS_TP = 2.451519356516061E+06              # Observer orbit time of periapsis JD TDB

SUN_A = 6.981597616379102E-03               # Sun orbit semi major axis
SUN_E = 2.694756164323808E-01               # Sun orbit eccentricity
SUN_I = np.radians(2.311817382515920E+01)   # Sun orbit inclination
SUN_OM = np.radians(3.852448313103996E+00)  # Sun orbit longitude of the ascending node
SUN_W = np.radians(3.231307573061241E+02)   # Sun orbit argument of periapsis
SUN_N = np.radians(8.293546641817494E-02)   # Sun orbit mean motion
SUN_TP = 2.452714888563351E+06              # Sun orbit time of periapsis JD TDB

# Variance of the observation direction (technically the inverse of the concentration parameter of the Fisher
# distribution)
DIR_VAR = 1E-6
# Variance of the visual magnitude
MAG_VAR = 1E-1


def real_position(t: float, a: float, e: float, i: float, om: float, w: float, n: float, tp: float) -> np.ndarray:
    mm = wrap_angle(n * (t - tp))
    ee = eccentric_anomaly_from_mean_anomaly(mm, e)
    v = true_anomaly_from_eccentric_anomaly(ee, e)
    r = distance_from_eccentric_anomaly(ee, a, e)
    return position_from_orbital_angles(om, w + v, i, r)


def gen_times(n: int, min_dist: float, max_dist: float, start: float, rng: np.random.Generator) -> list[float]:
    acc = start
    times = []
    for i in range(n):
        times.append(acc)
        acc += rng.uniform(min_dist, max_dist)
    return times


def observe(tgt_pos: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray, hh: float, gg: float,
            dir_var: float, vv_var: float, rng: np.random.Generator) -> tuple[float, float, float]:
    tgt_obs_pos = tgt_pos - obs_pos
    tgt_obs_dist = norm(tgt_obs_pos)

    tgt_obs_ray = tgt_obs_pos / tgt_obs_dist
    # Apply Fisher noise to the observation direction
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_fisher.html
    # noinspection PyTypeChecker
    tgt_obs_ray = sp.stats.vonmises_fisher(mu=tgt_obs_ray, kappa=(1.0 / dir_var), seed=rng).rvs(1)[0]
    ra, dec = direction_to_ra_dec(tgt_obs_ray)

    tgt_sun_dist = norm(tgt_pos - sun_pos)
    obs_sun_dist = norm(obs_pos - sun_pos)

    phi = phase_from_distances(tgt_sun_dist, tgt_obs_dist, obs_sun_dist)
    vv = visual_magnitude_from_absolute(hh, tgt_sun_dist, tgt_obs_dist, phi, gg)
    vv = rng.normal(loc=vv, scale=np.sqrt(vv_var))  # Apply Gaussian noise to the visual magnitude
    return ra, dec, vv


def prepare(obs: tuple[float, float, float]) -> np.ndarray:
    ra, dec, vv = obs
    return np.array([*angle_components(ra), *angle_components(dec), vv])


def main() -> None:
    rng = np.random.default_rng(0)
    times = gen_times(n=150, min_dist=0.01, max_dist=10.0, start=2451544.5, rng=rng)
    tgt_pos = [real_position(t, TGT_A, TGT_E, TGT_I, TGT_OM, TGT_W, TGT_N, TGT_TP) for t in times]
    obs_pos = [real_position(t, OBS_A, OBS_E, OBS_I, OBS_OM, OBS_W, OBS_N, OBS_TP) for t in times]
    sun_pos = [real_position(t, SUN_A, SUN_E, SUN_I, SUN_OM, SUN_W, SUN_N, SUN_TP) for t in times]

    a0 = TGT_A
    e0 = TGT_E
    i0 = TGT_I
    n0 = TGT_N
    hh0 = TGT_HH
    gg0 = TGT_GG
    mm0_hint = 0.0
    om0_hint = TGT_OM

    dir_var = DIR_VAR
    vv_var = MAG_VAR

    noisy = [observe(t, o, s, TGT_HH, TGT_GG, DIR_VAR, MAG_VAR, rng)
             for t, o, s in zip(tgt_pos, obs_pos, sun_pos)]
    prepared = [prepare(obs) for obs in noisy]

    naive = [reverse_transform(obs, o, s, hh0, gg0) for obs, o, s in zip(prepared, obs_pos, sun_pos)]

    points = MerweScaledSigmaPoints(13, alpha=1E-3, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=13, dim_z=5, dt=None, hx=measure, fx=propagate, points=points)

    x0 = initial_state(prepared[0], obs_pos[0], sun_pos[0], a0, e0, i0, n0, hh0, gg0, mm0_hint, om0_hint)
    pp0 = state_autocovariance_matrix(x0, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3)
    prev_t = times[0]
    ukf.x = x0
    ukf.P = pp0

    estimates = [finalize_transform(x0)]
    for t, obs, o, s in zip(times[1:], prepared[1:], obs_pos[1:], sun_pos[1:]):
        dt = t - prev_t
        ukf.Q = state_autocovariance_matrix(ukf.x, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8)
        ukf.predict(dt=dt)
        ukf.R = measurement_autocovariance_matrix(obs, dir_var, vv_var)
        ukf.update(z=obs, obs_pos=o, sun_pos=s)
        estimates.append(finalize_transform(ukf.x))
        prev_t = t

    from visualize import plot_errors_base, plot_3d_base
    plot_3d_base(obs_pos, tgt_pos, naive, estimates)
    plot_errors_base(tgt_pos, naive, estimates, times)  # , y_lim=0.05)


if __name__ == '__main__':
    main()
