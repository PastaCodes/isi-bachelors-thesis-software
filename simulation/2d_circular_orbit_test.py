import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from misc import wrap_angle_f, arctan2pos_f

'''
Suppose we want to monitor the polar coordinates (th and r) of a body in circular motion
Suppose we have noisy readings of these polar coordinates
The natural choice would be to consider [th, r] as both the observed vector and the state vector
This option is explored in the "direct" approach
The more stable version is to consider the vector [cos(th), sin(th), r],
where the first two components can also be thought of as the complex components of th
'''


REAL_RADIUS = 31.2
REAL_ANG_VEL = 0.243
REAL_TIME_OFFSET = 46.7
REAL_MEAS_NOISE_COV = np.diag([0.1, 0.05])  # R


def circle_orbit(t: float, time_offset: float, ang_vel: float, r: float) -> np.ndarray:
    th = wrap_angle_f(ang_vel * (t - time_offset))
    return np.array([th, r])


def gen_times(n: int, min_dist: float, max_dist: float, start: float, rng: np.random.Generator) -> list[float]:
    acc = start
    times = []
    for i in range(n):
        times.append(acc)
        acc += rng.uniform(min_dist, max_dist)
    return times


def disturb(obs: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.multivariate_normal(np.zeros_like(obs), cov)
    th, r = obs
    th_noise, r_noise = noise
    th_disturbed = wrap_angle_f(th + th_noise)
    r_disturbed = np.maximum(r + r_noise, 0)
    return np.array([th_disturbed, r_disturbed])


RNG = np.random.default_rng(0)
TIMES = gen_times(100, 0.0001, 0.05, 46.0, RNG)
TRUTH = [circle_orbit(t, REAL_TIME_OFFSET, REAL_ANG_VEL, REAL_RADIUS) for t in TIMES]
OBSS = [disturb(tr, REAL_MEAS_NOISE_COV, RNG) for tr in TRUTH]


def direct() -> None:

    def measurement_fn(state_vec: np.ndarray) -> np.ndarray:
        return state_vec  # The th,r coordinates are measured directly

    def transition_fn(before_vec: np.ndarray, dt: float, ang_vel: float) -> np.ndarray:
        th_before, r = before_vec
        th_after = wrap_angle_f(th_before + ang_vel * dt)
        return np.array([th_after, r])

    points = MerweScaledSigmaPoints(2, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=2, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = OBSS[0]
    pp0 = REAL_MEAS_NOISE_COV  # Since measurement is identity
    prev_t = TIMES[0]
    ukf.x = x0
    ukf.P = pp0
    ang_vel_guess = REAL_ANG_VEL

    estimates = [x0]
    for t, obs in zip(TIMES[1:], OBSS[1:]):
        ukf.Q = np.diag([0, 0])
        ukf.R = REAL_MEAS_NOISE_COV
        ukf.predict(dt=(t - prev_t), ang_vel=ang_vel_guess)
        ukf.update(z=obs)
        estimates.append(ukf.x)
        prev_t = t

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(TIMES, [tr[0] for tr in TRUTH], marker='o', c='g')
    ax1.scatter(TIMES, [obs[0] for obs in OBSS], marker='+', c='r')
    ax1.scatter(TIMES, [est[0] for est in estimates], marker='+', c='b')
    ax2.scatter(TIMES, [tr[1] for tr in TRUTH], marker='o', c='g')
    ax2.scatter(TIMES, [obs[1] for obs in OBSS], marker='+', c='r')
    ax2.scatter(TIMES, [est[1] for est in estimates], marker='+', c='b')
    plt.show()


def complex_repr() -> None:

    def convert_before(obs_vec: np.ndarray) -> np.ndarray:
        th, r = obs_vec
        cos = np.cos(th)
        sin = np.sin(th)
        return np.array([cos, sin, r])

    def convert_after(est_vec: np.ndarray) -> np.ndarray:
        cos, sin, r = est_vec
        th = arctan2pos_f(sin, cos)
        return np.array([th, r])

    converted_obss = [convert_before(obs) for obs in OBSS]

    def measurement_fn(state_vec: np.ndarray) -> np.ndarray:
        return state_vec

    def transition_fn(before_vec: np.ndarray, dt: float, ang_vel: float) -> np.ndarray:
        cos_before, sin_before, r = before_vec
        th_before = arctan2pos_f(sin_before, cos_before)
        th_after = wrap_angle_f(th_before + ang_vel * dt)
        cos_after = np.cos(th_after)
        sin_after = np.sin(th_after)
        return np.array([cos_after, sin_after, r])

    points = MerweScaledSigmaPoints(3, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=2, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = convert_before(OBSS[0])
    pp0 = np.diag([0.2, 0.2, 0.2])  # Technically should be a transformation of REAL_MEAS_NOISE_COV
    prev_t = TIMES[0]
    ukf.x = x0
    ukf.P = pp0
    ang_vel_guess = REAL_ANG_VEL

    estimates = []
    for t, obs in zip(TIMES[1:], converted_obss[1:]):
        ukf.Q = np.diag([0, 0, 0])
        ukf.R = np.diag([0.2, 0.2, 0.2])  # Technically should be a transformation of REAL_MEAS_NOISE_COV
        ukf.predict(dt=(t - prev_t), ang_vel=ang_vel_guess)
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


def compare() -> None:

    def measurement_fn(state_vec: np.ndarray) -> np.ndarray:
        return state_vec  # The th,r coordinates are measured directly

    def transition_fn(before_vec: np.ndarray, dt: float, ang_vel: float) -> np.ndarray:
        th_before, r = before_vec
        th_after = wrap_angle_f(th_before + ang_vel * dt)
        return np.array([th_after, r])

    points = MerweScaledSigmaPoints(2, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=2, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = OBSS[0]
    pp0 = REAL_MEAS_NOISE_COV  # Since measurement is identity
    prev_t = TIMES[0]
    ukf.x = x0
    ukf.P = pp0
    ang_vel_guess = REAL_ANG_VEL

    direct_estimates = [x0]
    for t, obs in zip(TIMES[1:], OBSS[1:]):
        ukf.Q = np.diag([0, 0])
        ukf.R = REAL_MEAS_NOISE_COV
        ukf.predict(dt=(t - prev_t), ang_vel=ang_vel_guess)
        ukf.update(z=obs)
        direct_estimates.append(ukf.x)
        prev_t = t

    def convert_before(obs_vec: np.ndarray) -> np.ndarray:
        th, r = obs_vec
        cos = np.cos(th)
        sin = np.sin(th)
        return np.array([cos, sin, r])

    def convert_after(est_vec: np.ndarray) -> np.ndarray:
        cos, sin, r = est_vec
        th = arctan2pos_f(sin, cos)
        return np.array([th, r])

    converted_obss = [convert_before(obs) for obs in OBSS]

    def measurement_fn(state_vec: np.ndarray) -> np.ndarray:
        return state_vec

    def transition_fn(before_vec: np.ndarray, dt: float, ang_vel: float) -> np.ndarray:
        cos_before, sin_before, r = before_vec
        th_before = arctan2pos_f(sin_before, cos_before)
        th_after = wrap_angle_f(th_before + ang_vel * dt)
        cos_after = np.cos(th_after)
        sin_after = np.sin(th_after)
        return np.array([cos_after, sin_after, r])

    points = MerweScaledSigmaPoints(3, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=3, dim_z=2, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = convert_before(OBSS[0])
    pp0 = np.diag([0.2, 0.2, 0.2])  # Technically should be a transformation of REAL_MEAS_NOISE_COV
    prev_t = TIMES[0]
    ukf.x = x0
    ukf.P = pp0
    ang_vel_guess = REAL_ANG_VEL

    complex_estimates = []
    for t, obs in zip(TIMES[1:], converted_obss[1:]):
        ukf.Q = np.diag([0, 0, 0])
        ukf.R = np.diag([0.2, 0.2, 0.2])  # Technically should be a transformation of REAL_MEAS_NOISE_COV
        ukf.predict(dt=(t - prev_t), ang_vel=ang_vel_guess)
        ukf.update(z=obs)
        complex_estimates.append(ukf.x)
        prev_t = t

    converted = [OBSS[0]] + [convert_after(est) for est in complex_estimates]

    import matplotlib.pyplot as plt
    plt.xlabel('tempo')
    plt.ylabel('angolo')
    plt.scatter(TIMES, [tr[0] for tr in TRUTH], marker='o', c='g', s=15, label='Valore reale')
    plt.scatter(TIMES, [obs[0] for obs in OBSS], marker='+', c='r', label='Misurazione')
    plt.scatter(TIMES, [est[0] for est in direct_estimates], marker='^', c='c', s=15, label='Stima diretta')
    plt.scatter(TIMES, [est[0] for est in converted], marker='s', c='b', s=15, label='Stima indiretta')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    compare()
