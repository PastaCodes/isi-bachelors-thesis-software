import numpy as np
from matplotlib import pyplot as plt

from filter import kf_estimate
from main import BODY_BENNU
from model import naive_transform
from parse import parse_observations, parse_ephemeris


def extract_xyz(array: list | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.array([pos[0] for pos in array])
    yy = np.array([pos[1] for pos in array])
    zz = np.array([pos[2] for pos in array])
    return xx, yy, zz


def cherry_picked() -> None:
    obss = list(parse_observations(BODY_BENNU))
    ephs = list(parse_ephemeris(BODY_BENNU))
    ests = list(kf_estimate(obss[:130],
                            a0=1.128,
                            e0=0.204,
                            i0=0.514,
                            mm0_hint=6.28,
                            om0_hint=0.0,
                            mm_var=3.5E-7,
                            a_var=2.9E-8,
                            e_var=1.0E-8,
                            i_var=1.6E-10,
                            om_var=5.0E-11,
                            w_var=1.7E-7,
                            n_var=1.1E-11,
                            dir_var=1.1E-4,
                            vv_var=2.7E-1,
                            dt_exp=1.22))
    obss_pos = [eph.observer_position for eph in ephs[:130]]
    ephs_pos = [eph.target_position for eph in ephs[:130]]
    naive_pos = [naive_transform(obs, eph) for obs, eph in zip(obss[:130], ephs[:130])]
    ests_pos = [est.position for est in ests]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0.878, 1.128)
    ax.set_ylim(-0.2, 0.05)
    ax.set_zlim(-0.173, 0.077)
    ax.scatter(*extract_xyz(obss_pos), marker='D', c='k', depthshade=False, label='Osservatore')
    ax.scatter(*extract_xyz(ephs_pos), marker='o', c='g', depthshade=False, label='Posizione reale')
    ax.scatter(*extract_xyz(naive_pos), marker='+', c='r', depthshade=False, label='Approccio naïve')
    ax.scatter(*extract_xyz(ests_pos), marker='s', c='b', depthshade=False, label='Stima UKF')
    ax.legend()
    plt.show()
