import numpy as np
import scipy as sp


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
