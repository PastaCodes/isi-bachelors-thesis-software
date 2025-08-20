import numpy as np
from astropy.time import TimeDelta, Time
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from classes import MinorPlanetState, MinorPlanet, Observatory
from model import reverse_transform, propagate, measure_transform
from parse import parse_observations


def do_filter(body: MinorPlanet):
    obss = parse_observations(body)

    def measurement_fn(state_vec: np.ndarray, epoch: Time, observatory: Observatory) -> np.ndarray:
        state = MinorPlanetState.from_vector(state_vec, body, epoch)
        measurement = measure_transform(state, observatory)
        return measurement.to_vector()

    def transition_fn(before_vec: np.ndarray, dt: TimeDelta, before_epoch: Time) -> np.ndarray:
        before = MinorPlanetState.from_vector(before_vec, body, before_epoch)
        after = propagate(before, delta_time=dt)
        return after.to_vector()

    points = MerweScaledSigmaPoints(9, alpha=.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=9, dim_z=5, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = reverse_transform(obss[0], body.initial_state_hint).to_vector()
    pp0 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    prev_epoch = obss[0].epoch
    ukf.x = x0
    ukf.P = pp0

    for i, obs in enumerate(obss[1:]):
        ukf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ukf.R = np.diag([0.001, 0.001, 0.001, 0.001, 0.1])
        ukf.predict(dt=(obs.epoch - prev_epoch), before_epoch=prev_epoch)
        ukf.update(z=obs.to_vector(), epoch=obs.epoch, observatory=obs.observatory)
        state_estimate = MinorPlanetState.from_vector(ukf.x, body, obs.epoch)
        print(state_estimate.position)
        input()
        prev_epoch = obs.epoch
