import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from classes import MinorPlanet
from misc import get_sun_position, get_location_position
from model import propagate, measure, finalize_transform, initial_state
from orbit import get_mean_motion
from parse import parse_observations


def do_filter(body: MinorPlanet, dir_conc: float, vv_var: float,
              a0: float = 1.0, e0: float = 0.5, i0: float = 0.25 * np.pi, n0: float | None = None,
              hh0: float = 20.0, gg0: float = 0.15, mm0_hint: float = 0.0, om0_hint: float = 0.0) -> None:
    obss = parse_observations(body)

    points = MerweScaledSigmaPoints(n=13, alpha=.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=13, dim_z=5, dt=None, hx=measure, fx=propagate, points=points)

    if n0 is None:
        n0 = get_mean_motion(a0)

    obs0 = obss[0]
    t0 = obs0.epoch
    x0 = initial_state(obs0.to_vector(),
                       get_location_position(obs0.observatory.location, t0),
                       get_sun_position(t0),
                       a0, e0, i0, n0, hh0, gg0, mm0_hint, om0_hint)
    pp0 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    prev_epoch = t0
    ukf.x = x0
    ukf.P = pp0

    for i, obs in enumerate(obss[1:]):
        obs_vec = obs.to_vector()
        cos_ra, sin_ra, cos_dec, sin_dec, vv = obs_vec
        dt = (obs.epoch - prev_epoch).to_value('jd')
        ukf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ukf.R = np.array([[sin_ra * sin_ra / (dir_conc * cos_dec * cos_dec),
                           -sin_ra * cos_ra / (dir_conc * cos_dec * cos_dec),
                           0.0, 0.0, 0.0],
                          [-sin_ra * cos_ra / (dir_conc * cos_dec * cos_dec),
                           cos_ra * cos_ra / (dir_conc * cos_dec * cos_dec),
                           0.0, 0.0, 0.0],
                          [0.0, 0.0,
                           sin_dec * sin_dec / dir_conc,
                           -sin_dec * cos_dec / dir_conc,
                           0.0],
                          [0.0, 0.0,
                           -sin_dec * cos_dec / dir_conc,
                           cos_dec * cos_dec / dir_conc,
                           0.0],
                          [0.0, 0.0, 0.0, 0.0, vv_var]])
        ukf.predict(dt=dt)
        sun_pos = get_sun_position(obs.epoch)
        obs_pos = get_location_position(obs.observatory.location, obs.epoch)
        ukf.update(z=obs_vec, obs_pos=obs_pos, sun_pos=sun_pos)
        print(finalize_transform(ukf.x))
        input()
        prev_epoch = obs.epoch
