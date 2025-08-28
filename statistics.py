import numpy as np
import scipy as sp

from classes import MinorPlanet
from model import propagate, finalize_transform
from parse import parse_ephemeris, parse_observations


def fisher_ra_dec_variance_test(ra0: float, dec0: float, kappa: float, n: float, rng: np.random.Generator) -> None:
    mu = np.array([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
    # noinspection PyTypeChecker
    fisher = sp.stats.vonmises_fisher(mu=mu, kappa=kappa, seed=rng)
    # noinspection PyTypeChecker
    sampled = fisher.rvs(size=n)

    sampled_ra_dec = [[np.arctan2(xyz[1], xyz[0]), np.arcsin(xyz[2])] for xyz in sampled]
    cov = np.cov(sampled_ra_dec, rowvar=False, bias=True)
    print('Computed:')
    print(cov)
    cov = np.diag([np.pow(np.cos(dec0), -2), 1.0]) / kappa
    print('Expected:')
    print(cov)


def unit_circle_components_variance_test(th0: float, kappa: float, n: float, rng: np.random.Generator) -> None:
    # noinspection PyTypeChecker
    sampled = sp.stats.vonmises.rvs(kappa=kappa, loc=th0, size=n, random_state=rng)
    sampled_cos_sin = [[np.cos(th), np.sin(th)] for th in sampled]
    cov = np.cov(sampled_cos_sin, rowvar=False, bias=True)
    print('Computed:')
    print(cov)
    cov = np.array([[np.sin(th0)*np.sin(th0), -np.sin(th0)*np.cos(th0)],
                    [-np.sin(th0)*np.cos(th0), np.cos(th0)*np.cos(th0)]]) / kappa
    print('Expected:')
    print(cov)


def time_dependence_estimation(body: MinorPlanet) -> float:
    ephs = list(parse_ephemeris(body))
    eph0 = ephs[0]
    x0 = eph0.to_state_vector()
    t0 = eph0.epoch.tdb.to_value('jd')

    dts = []
    traces = []
    for eph in ephs[1:]:
        epoch = eph.epoch.tdb.to_value('jd')
        dt = epoch - t0
        dts.append(dt)
        x = propagate(x0, dt)
        calc = finalize_transform(x)
        real = eph.target_position
        err = np.vstack(calc - real)
        cov = err @ err.T
        traces.append(np.trace(cov))

    (a, b), _ = sp.optimize.curve_fit(lambda _x, _a, _b: _a * _x ** _b, dts, traces)
    xx = np.linspace(0, dts[-1], 50)
    yy = a * xx ** b

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xx, yy)
    plt.scatter(dts, traces)
    plt.show()

    return b


def extract_angle_covariance(mat: np.ndarray, cos: float, sin: float) -> np.ndarray:
    return np.divide(mat, np.array([[sin * sin, -sin * cos],
                                    [-sin * cos, cos * cos]]))


def process_covariance_estimation(body: MinorPlanet,
                                  dt_exp: float) -> tuple[float, float, float, float, float, float, float]:
    ephs = list(parse_ephemeris(body))
    prev_x = ephs[0].to_state_vector()
    prev_epoch = ephs[0].epoch.tdb.to_value('jd')

    mm_covs = []
    a_vars = []
    e_vars = []
    i_covs = []
    om_covs = []
    w_covs = []
    n_vars = []
    for eph in ephs[1:]:
        epoch = eph.epoch.tdb.to_value('jd')
        dt = epoch - prev_epoch
        calc = propagate(prev_x, dt)
        real = eph.to_state_vector()
        err = np.vstack(calc - real)
        cov = err @ err.T
        corrected_cov = np.pow(dt, -dt_exp) * cov
        cos_mm, sin_mm, _, _, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, _, _, _ = calc
        mm_covs.append(extract_angle_covariance(corrected_cov[0:2, 0:2], cos_mm, sin_mm))
        a_vars.append(corrected_cov[2, 2])
        e_vars.append(corrected_cov[3, 3])
        i_covs.append(extract_angle_covariance(corrected_cov[4:6, 4:6], cos_i, sin_i))
        om_covs.append(extract_angle_covariance(corrected_cov[6:8, 6:8], cos_om, sin_om))
        w_covs.append(extract_angle_covariance(corrected_cov[8:10, 8:10], cos_w, sin_w))
        n_vars.append(corrected_cov[10, 10])
        prev_x = real
        prev_epoch = epoch

    mm_var = np.mean(np.array(mm_covs[1:]).flatten())
    a_var = np.mean(a_vars[1:])
    e_var = np.mean(e_vars[1:])
    i_var = np.mean(np.array(i_covs[1:]).flatten())
    om_var = np.mean(np.array(om_covs[1:]).flatten())
    w_var = np.mean(np.array(w_covs[1:]).flatten())
    n_var = np.mean(n_vars[1:])
    # noinspection PyTypeChecker
    return mm_var, a_var, e_var, i_var, om_var, w_var, n_var


def extract_super_covariance(mat1: np.ndarray, mat2: np.ndarray, cos_ra: float, sin_ra: float,
                             cos_dec: float, sin_dec: float) -> tuple[np.ndarray, np.ndarray]:
    return (np.divide(mat1, np.array([[sin_ra * sin_ra, -sin_ra * cos_ra],
                                      [-sin_ra * cos_ra, cos_ra * cos_ra]]) / (cos_dec * cos_dec)),
            np.divide(mat2, np.array([[sin_dec * sin_dec, -sin_dec * cos_dec],
                                      [-sin_dec * cos_dec, cos_dec * cos_dec]])))


def measurement_covariance_estimation(body: MinorPlanet) -> tuple[float, float]:
    ephs = parse_ephemeris(body)
    obss = parse_observations(body)

    dir_covs = []
    vv_vars = []
    for obs, eph in zip(obss, ephs):
        measured = obs.to_vector()
        real = eph.to_geometric_measurement_vector()
        err = np.vstack(measured - real)
        cov = err @ err.T
        cos_ra, sin_ra, cos_dec, sin_dec, _ = measured
        mat1, mat2 = extract_super_covariance(cov[0:2, 0:2], cov[2:4, 2:4], cos_ra, sin_ra, cos_dec, sin_dec)
        dir_covs.extend([mat1, mat2])
        vv_vars.append(cov[4, 4])

    dir_var = np.mean(np.array(dir_covs[1:]).flatten())
    vv_var = np.mean(vv_vars[1:])
    # noinspection PyTypeChecker
    return dir_var, vv_var
