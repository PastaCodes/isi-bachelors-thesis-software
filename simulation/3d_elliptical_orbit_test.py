import numpy as np
import scipy as sp
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from misc import wrap_angle_f, arctan2pos_f


REAL_SEMI_MAJOR_AXIS = 2.516
REAL_ECCENTRICITY = 0.662
REAL_INCLINATION = 0.469
REAL_ASC_LONG = 2.31
REAL_PERI_ARG = 0.26
REAL_TIME_OFFSET = 46.7
REAL_MEAS_NOISE_COV = np.diag([0.01, 0.05])  # R


def get_mean_motion(gravity: float, semi_major_axis: float) -> float:
    return np.sqrt(gravity / semi_major_axis ** 3)


def get_beta(eccentricity: float) -> float:
    return eccentricity / (1 + np.sqrt(1 - eccentricity * eccentricity))


REAL_MEAN_MOTION = get_mean_motion(39.476, REAL_SEMI_MAJOR_AXIS)
REAL_BETA = get_beta(REAL_ECCENTRICITY)


def real_orbit(t: float) -> np.ndarray:
    mean_anom = wrap_angle_f(REAL_MEAN_MOTION * (t - REAL_TIME_OFFSET))

    def f(ee: float) -> float:
        return ee - REAL_ECCENTRICITY * np.sin(ee) - mean_anom

    def f_prime(ee: float) -> float:
        return 1 - REAL_ECCENTRICITY * np.cos(ee)

    ecc_anom = sp.optimize.newton(f, mean_anom, f_prime)
    true_anom = ecc_anom + 2 * np.atan(REAL_BETA * np.sin(ecc_anom) / (1 - REAL_BETA * np.cos(ecc_anom)))
    dist = REAL_SEMI_MAJOR_AXIS * (1 - REAL_ECCENTRICITY * np.cos(ecc_anom))
    return np.array([true_anom, dist])


def gen_times(n: int, min_dist: float, max_dist: float, start: float, rng: np.random.Generator) -> list[float]:
    acc = start
    times = []
    for i in range(n):
        times.append(acc)
        acc += rng.uniform(min_dist, max_dist)
    return times


def disturb(obs: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.multivariate_normal(np.zeros_like(obs), cov)
    v, r = obs
    v_noise, r_noise = noise
    v_disturbed = wrap_angle_f(v + v_noise)
    r_disturbed = np.maximum(r + r_noise, 0)
    return np.array([v_disturbed, r_disturbed])


RNG = np.random.default_rng(0)
TIMES = gen_times(150, 0.0001, 0.05, 2000.0, RNG)
TRUTH = [real_orbit(t) for t in TIMES]
OBSS = [disturb(tr, REAL_MEAS_NOISE_COV, RNG) for tr in TRUTH]


def main() -> None:
    a = REAL_SEMI_MAJOR_AXIS
    e = REAL_ECCENTRICITY
    n = REAL_MEAN_MOTION
    b = REAL_BETA

    def convert_before(obs_vec: np.ndarray) -> np.ndarray:
        v, r = obs_vec
        cos_v = np.cos(v)
        sin_v = np.sin(v)
        return np.array([cos_v, sin_v, r])

    def convert_after(est_vec: np.ndarray) -> np.ndarray:
        cos_v, sin_v, r = est_vec
        v = arctan2pos_f(sin_v, cos_v)
        return np.array([v, r])

    converted_obss = [convert_before(obs) for obs in OBSS]

    def measurement_fn(state_vec: np.ndarray) -> np.ndarray:
        return state_vec

    def transition_fn(before_vec: np.ndarray, dt: float) -> np.ndarray:
        cos_v_before, sin_v_before, r_before = before_vec
        ee_before = arctan2pos_f(np.sqrt(1 - e * e) * sin_v_before, e + cos_v_before)
        mm_before = ee_before - e * np.sin(ee_before)
        mm_after = wrap_angle_f(mm_before + n * dt)

        def f(ee: float) -> float:
            return ee - e * np.sin(ee) - mm_before

        def f_prime(ee: float) -> float:
            return 1 - e * np.cos(ee)

        ee_after = sp.optimize.newton(f, mm_after, f_prime)
        v_after = ee_after + 2 * np.atan(b * np.sin(ee_after) / (1 - b * np.cos(ee_after)))
        r_after = a * (1 - e * np.cos(ee_after))
        cos_v_after = np.cos(v_after)
        sin_v_after = np.sin(v_after)
        return np.array([cos_v_after, sin_v_after, r_after])

    points = MerweScaledSigmaPoints(3, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=3, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = convert_before(OBSS[0])
    pp0 = np.diag([0.5, 0.5, 0.5])
    prev_t = TIMES[0]
    ukf.x = x0
    ukf.P = pp0

    estimates = []
    for t, obs in zip(TIMES[1:], converted_obss[1:]):
        ukf.Q = 0.01 * np.eye(3)
        ukf.R = np.diag([0.05, 0.05, 0.1])
        ukf.predict(dt=(t - prev_t))
        ukf.update(z=obs)
        estimates.append(ukf.x)
        prev_t = t

    converted = [OBSS[0]] + [convert_after(est) for est in estimates]

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(TIMES, [tr[0] for tr in TRUTH], marker='o', c='g')
    ax1.scatter(TIMES, [obs[0] for obs in OBSS], marker='+', c='r')
    ax1.scatter(TIMES, [est[0] for est in converted], marker='+', c='b')
    ax2.scatter(TIMES, [tr[1] for tr in TRUTH], marker='o', c='g')
    ax2.scatter(TIMES, [obs[1] for obs in OBSS], marker='+', c='r')
    ax2.scatter(TIMES, [est[1] for est in converted], marker='+', c='b')
    plt.show()


if __name__ == '__main__':
    main()
