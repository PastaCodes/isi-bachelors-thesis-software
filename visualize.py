import numpy as np
from matplotlib import pyplot as plt

from classes import MinorPlanet
from filter import kf_estimate
from bodies import BODY_BENNU, BODY_1950_DA, BODY_MJOLNIR, BODY_CASTALIA
from misc import norm
from model import naive_transform
from parse import parse_observations, parse_ephemeris


def plot_errors_base(eph_pos: list[np.ndarray] | np.ndarray, naive_pos: list[np.ndarray] | np.ndarray,
                     est_pos: list[np.ndarray] | np.ndarray, tt: list[float] | np.ndarray,
                     use_time: bool = True, y_lim: float | None = None) -> None:
    if use_time:
        plt.xlabel('t')
    else:
        # plt.xticks(range(tt[0], tt[-1] + 1))
        plt.xlabel('passo')
    plt.ylabel('Errore sulla posizione')
    plt.axhline(c='k', lw=0.5)
    plt.scatter(tt, [norm(n - truth) for n, truth in zip(naive_pos, eph_pos)], marker='+', c='r',
                label='Approccio diretto')
    plt.scatter(tt, [norm(est - truth) for est, truth in zip(est_pos, eph_pos)], marker='s', c='b',
                label='Stima UKF')
    if y_lim:
        plt.ylim(-0.05 * y_lim, 1.05 * y_lim)
    plt.legend()
    plt.show()


def plot_errors(body: MinorPlanet, compute_from: int = 0, display_from: int = None, to: int = None,
                use_time: bool = True, y_lim: float | None = None) -> None:
    if display_from is None:
        display_from = compute_from
    else:
        assert display_from >= compute_from
    obss = list(parse_observations(body))
    ephs = list(parse_ephemeris(body))
    if to is None:
        to = len(ephs)
    ests = list(kf_estimate(obss[compute_from:to]))
    eph_pos = [eph.target_position for eph in ephs[display_from:to]]
    naive_pos = [naive_transform(obs, eph) for obs, eph in zip(obss[display_from:to], ephs[display_from:to])]
    est_pos = [est.position for est in ests[display_from - compute_from:]]
    tt = [eph.epoch.to_value('jd') for eph in ephs[display_from:to]] if use_time else range(display_from, to)
    plot_errors_base(eph_pos, naive_pos, est_pos, tt, use_time, y_lim)


def extract_xyz(array: list | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.array([pos[0] for pos in array])
    yy = np.array([pos[1] for pos in array])
    zz = np.array([pos[2] for pos in array])
    return xx, yy, zz


# noinspection PyUnresolvedReferences
def plot_3d_base(obs_pos: list[np.ndarray] | np.ndarray, eph_pos: list[np.ndarray] | np.ndarray,
                 naive_pos: list[np.ndarray] | np.ndarray, est_pos: list[np.ndarray] | np.ndarray) -> None:
    obs_xx, obs_yy, obs_zz = extract_xyz(obs_pos)
    eph_xx, eph_yy, eph_zz = extract_xyz(eph_pos)
    naive_xx, naive_yy, naive_zz = extract_xyz(naive_pos)
    est_xx, est_yy, est_zz = extract_xyz(est_pos)
    min_x = np.min(np.concatenate([obs_xx, eph_xx, naive_xx, est_xx]))
    max_x = np.max(np.concatenate([obs_xx, eph_xx, naive_xx, est_xx]))
    min_y = np.min(np.concatenate([obs_yy, eph_yy, naive_yy, est_yy]))
    max_y = np.max(np.concatenate([obs_yy, eph_yy, naive_yy, est_yy]))
    min_z = np.min(np.concatenate([obs_zz, eph_zz, naive_zz, est_zz]))
    max_z = np.max(np.concatenate([obs_zz, eph_zz, naive_zz, est_zz]))
    size = 1.1 * np.max([max_x - min_x, max_y - min_y, max_z - min_z])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim((min_x + max_x - size) / 2, (min_x + max_x + size) / 2)
    ax.set_ylim((min_y + max_y - size) / 2, (min_y + max_y + size) / 2)
    ax.set_zlim((min_z + max_z - size) / 2, (min_z + max_z + size) / 2)
    ax.scatter(obs_xx, obs_yy, obs_zz, marker='D', c='k', depthshade=False, label='Osservatore')
    ax.scatter(eph_xx, eph_yy, eph_zz, marker='o', c='g', depthshade=False, label='Posizione reale')
    ax.scatter(naive_xx, naive_yy, naive_zz, marker='+', c='r', depthshade=False, label='Approccio diretto')
    ax.scatter(est_xx, est_yy, est_zz, marker='s', c='b', depthshade=False, label='Stima UKF')
    ax.legend()
    plt.show()


def plot_3d(body: MinorPlanet, compute_from: int = 0, display_from: int = None, to: int = None) -> None:
    if display_from is None:
        display_from = compute_from
    else:
        assert display_from >= compute_from
    obss = list(parse_observations(body))
    ephs = list(parse_ephemeris(body))
    if to is None:
        to = len(ephs)
    ests = list(kf_estimate(obss[compute_from:to]))
    obs_pos = [eph.observer_position for eph in ephs[display_from:to]]
    eph_pos = [eph.target_position for eph in ephs[display_from:to]]
    naive_pos = [naive_transform(obs, eph) for obs, eph in zip(obss[display_from:to], ephs[display_from:to])]
    est_pos = [est.position for est in ests[display_from - compute_from:]]
    plot_3d_base(obs_pos, eph_pos, naive_pos, est_pos)


def plot_3d_bennu() -> None:
    plot_3d(BODY_BENNU, to=130)


def plot_errors_bennu() -> None:
    plot_errors(BODY_BENNU, to=130, use_time=False)


def plot_3d_mjolnir() -> None:
    plot_3d(BODY_MJOLNIR, to=58)


def plot_errors_mjolnir() -> None:
    plot_errors(BODY_MJOLNIR, to=58, use_time=False)


def plot_3d_1950da() -> None:
    plot_3d(BODY_1950_DA, to=69)


def plot_errors_1950da() -> None:
    plot_errors(BODY_1950_DA, to=69, use_time=False)


def plot_3d_1950da_2() -> None:
    plot_3d(BODY_1950_DA, compute_from=200, to=250)


def plot_errors_1950da_2() -> None:
    plot_errors(BODY_1950_DA, compute_from=200, to=250, use_time=False)


def plot_3d_castalia() -> None:
    plot_3d(BODY_CASTALIA, to=50)


def plot_errors_castalia() -> None:
    plot_errors(BODY_CASTALIA, to=50, use_time=False)
