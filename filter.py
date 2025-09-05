from typing import Iterable

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from classes import MinorPlanetEstimate, MinorPlanetObservation
from model import propagate, measure, initial_state, state_autocovariance_matrix, measurement_autocovariance_matrix
from sky import get_sun_position, get_location_position


def kf_estimate(obss: list[MinorPlanetObservation]) -> Iterable[MinorPlanetEstimate]:
    body = obss[0].target_body

    points = MerweScaledSigmaPoints(n=13, alpha=.1, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=13, dim_z=5, dt=None, hx=measure, fx=propagate, points=points)

    obs0 = obss[0]
    t0 = obs0.epoch
    x0 = initial_state(obs0.to_vector(),
                       get_location_position(obs0.observatory.location, t0),
                       get_sun_position(t0),
                       body.a0, body.e0, body.i0, body.n0, body.hh0, body.gg0, body.mm0_hint, body.om0_hint)
    yield MinorPlanetEstimate.from_state_vector(x0, body, t0)
    pp0 = state_autocovariance_matrix(x0, body.mm0_var, body.a0_var, body.e0_var, body.i0_var,
                                      body.om0_var, body.w0_var, body.n0_var, body.hh0_var, body.gg0_var)
    prev_epoch = t0.tdb
    ukf.x = x0
    ukf.P = pp0

    for i, obs in enumerate(obss[1:]):
        obs_vec = obs.to_vector()
        epoch = obs.epoch.tdb
        dt = (epoch - prev_epoch).to_value('jd')
        ukf.Q = (state_autocovariance_matrix(ukf.x, body.mm_var, body.a_var, body.e_var, body.i_var,
                                             body.om_var, body.w_var, body.n_var, body.hh_var, body.gg_var)
                 * np.pow(dt, body.dt_exp))
        ukf.predict(dt=dt)
        sun_pos = get_sun_position(epoch)
        obs_pos = get_location_position(obs.observatory.location, epoch)
        ukf.R = measurement_autocovariance_matrix(obs_vec, body.dir_var, body.vv_var)
        ukf.update(z=obs_vec, obs_pos=obs_pos, sun_pos=sun_pos)
        yield MinorPlanetEstimate.from_state_vector(ukf.x, body, epoch)
        prev_epoch = epoch
