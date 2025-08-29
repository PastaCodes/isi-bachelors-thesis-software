from typing import Iterable

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from classes import MinorPlanetEstimate, MinorPlanetObservation
from model import propagate, measure, initial_state, state_autocovariance_matrix, measurement_autocovariance_matrix
from orbit import get_mean_motion
from sky import get_sun_position, get_location_position


def kf_estimate(obss: list[MinorPlanetObservation],
                a0: float = 1.0, e0: float = 0.5, i0: float = 0.25 * np.pi, n0: float | None = None,
                hh0: float = None, gg0: float = None, mm0_hint: float = 0.0, om0_hint: float = 0.0,
                mm0_var: float = 1E-3, a0_var: float = 1E-5, e0_var: float = 1E-5, i0_var: float = 1E-5,
                om0_var: float = 1E-3, w0_var: float = 1E-3, n0_var: float = 1E-3,
                hh0_var: float = 1E-5, gg0_var: float = 1E-3,
                mm_var: float = 1E-5, a_var: float = 1E-7, e_var: float = 1E-7, i_var: float = 1E-7,
                om_var: float = 1E-7, w_var: float = 1E-7, n_var: float = 1E-7,
                hh_var: float = 1E-12, gg_var: float = 1E-12, dir_var: float = 1E-5, vv_var: float = 1E-2,
                dt_exp: float = 1.0) -> Iterable[MinorPlanetEstimate]:
    body = obss[0].target_body
    if hh0 is None:
        hh0 = body.absolute_magnitude
    if gg0 is None:
        gg0 = body.slope_parameter

    points = MerweScaledSigmaPoints(n=13, alpha=.1, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=13, dim_z=5, dt=None, hx=measure, fx=propagate, points=points)

    if n0 is None:
        n0 = get_mean_motion(a0)

    obs0 = obss[0]
    t0 = obs0.epoch
    x0 = initial_state(obs0.to_vector(),
                       get_location_position(obs0.observatory.location, t0),
                       get_sun_position(t0),
                       a0, e0, i0, n0, hh0, gg0, mm0_hint, om0_hint)
    yield MinorPlanetEstimate.from_state_vector(x0, body, t0)
    pp0 = state_autocovariance_matrix(x0, mm0_var, a0_var, e0_var, i0_var, om0_var, w0_var, n0_var, hh0_var, gg0_var)
    prev_epoch = t0.tdb
    ukf.x = x0
    ukf.P = pp0

    for i, obs in enumerate(obss[1:]):
        obs_vec = obs.to_vector()
        epoch = obs.epoch.tdb
        dt = (epoch - prev_epoch).to_value('jd')
        ukf.Q = (state_autocovariance_matrix(ukf.x, mm_var, a_var, e_var, i_var, om_var, w_var, n_var, hh_var, gg_var)
                 * np.pow(dt, dt_exp))
        ukf.predict(dt=dt)
        sun_pos = get_sun_position(epoch)
        obs_pos = get_location_position(obs.observatory.location, epoch)
        ukf.R = measurement_autocovariance_matrix(obs_vec, dir_var, vv_var)
        ukf.update(z=obs_vec, obs_pos=obs_pos, sun_pos=sun_pos)
        yield MinorPlanetEstimate.from_state_vector(ukf.x, body, epoch)
        prev_epoch = epoch
