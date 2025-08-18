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

    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=None, hx=measurement_fn, fx=transition_fn, points=points)

    x0 = reverse_transform(obss[0], body.initial_state_hint).to_vector()
    pp0 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    epoch = obss[0].epoch
    ukf.x = x0
    ukf.P = pp0

    for i, obs in enumerate(obss[1:]):
        ukf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ukf.R = np.diag([0.1, 0.1, 0.1])
        ukf.predict(dt=(obs.epoch - epoch), before_epoch=epoch)
        ukf.update(z=obs.to_vector(), epoch=obs.epoch, observatory=obs.observatory)
        state = MinorPlanetState.from_vector(ukf.x, body, obs.epoch)
        print(state.position)
        input()
        epoch = obs.epoch
