import astropy.units as u
import numpy as np
from astropy.time import TimeDelta
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from classes import MinorPlanet, StateHint, MinorPlanetState
from filter import do_filter
from model import reverse_transform, propagate, measure_transform

BODY_BENNU = MinorPlanet(name='1999 RQ36',
                         hh=(20.21 * u.mag),
                         gg=(-0.031 * u.dimensionless_unscaled),
                         a=(1.128 * u.au),
                         e=(0.204 * u.dimensionless_unscaled),
                         i=(0.514 * u.rad),
                         observations_filepath='data/Bennu.txt',
                         ephemeris_filepath='data/Bennu_eph.txt',
                         initial_state_hint=StateHint(mean_anomaly=(2 * np.pi * u.rad),
                                                      ascending_longitude=(0 * u.rad)))

BODY_2010_RF12 = MinorPlanet(name='2010 RF12',
                             hh=(28.42 * u.mag),
                             gg=(0.15 * u.dimensionless_unscaled),
                             a=(1.061 * u.au),
                             e=(0.188 * u.dimensionless_unscaled),
                             i=(0.424 * u.rad),
                             observations_filepath='data/2010_rf12.txt',
                             ephemeris_filepath='data/2010_rf12_eph.txt')

BODY_2024_YR4 = MinorPlanet(name='2024 YR4',
                            hh=(23.92 * u.mag),
                            gg=(0.15 * u.dimensionless_unscaled),
                            a=(2.516 * u.au),
                            e=(0.662 * u.dimensionless_unscaled),
                            i=(0.469 * u.rad),
                            observations_filepath='data/2024_yr4.txt',
                            ephemeris_filepath='data/2024_yr4_eph.txt')


if __name__ == '__main__':
    do_filter(BODY_BENNU)
    exit(0)

    from parse import parse_observations, parse_ephemeris
    obss = parse_observations(BODY_BENNU)
    ephs = list(parse_ephemeris(BODY_BENNU).values())

    body = BODY_BENNU
    obs0 = obss[0]
    eph0 = ephs[0]
    print(f'Measured k=0: ra={obs0.right_ascension.to(u.rad)}, dec={obs0.declination.to(u.rad)}, vmag={obs0.visual_magnitude.to(u.mag)}')
    print(f'Ephemeris (observ) k=0: ra={eph0.right_ascension.to(u.rad)}, dec={eph0.declination.to(u.rad)}, vmag={eph0.visual_magnitude.to(u.mag)}')
    state0_calc = reverse_transform(obs0, StateHint(eph0.mean_anomaly, eph0.ascending_longitude))
    print(f'Calculated state k=0: pos={state0_calc.position.get_xyz().to(u.au)}, mm={state0_calc.mean_anomaly.to(u.rad)}, om={state0_calc.ascending_longitude.to(u.rad)}, w={state0_calc.periapsis_argument.to(u.rad)}')
    print(f'Ephemeris (state) k=0: pos={eph0.position.get_xyz().to(u.au)}, mm={eph0.mean_anomaly.to(u.rad)}, om={eph0.ascending_longitude.to(u.rad)}, w={eph0.periapsis_argument.to(u.rad)}')
    eph1 = ephs[1]
    state1_pred = propagate(state0_calc, eph1.epoch)
    print(f'Predicted state k=1: pos={state1_pred.position.get_xyz().to(u.au)}, mm={state1_pred.mean_anomaly.to(u.rad)}, om={state1_pred.ascending_longitude.to(u.rad)}, w={state1_pred.periapsis_argument.to(u.rad)}')
    print(f'Ephemeris (state) k=1: pos={eph1.position.get_xyz().to(u.au)}, mm={eph1.mean_anomaly.to(u.rad)}, om={eph1.ascending_longitude.to(u.rad)}, w={eph1.periapsis_argument.to(u.rad)}')
    obs1 = obss[1]
    print(f'Measured k=1: ra={obs1.right_ascension.to(u.rad)}, dec={obs1.declination.to(u.rad)}, vmag={obs1.visual_magnitude.to(u.mag)}')
    state1_calc = reverse_transform(obs1, StateHint(eph1.mean_anomaly, eph1.ascending_longitude))
    print(f'Calculated state k=1: pos={state1_calc.position.get_xyz().to(u.au)}, mm={state1_calc.mean_anomaly.to(u.rad)}, om={state1_calc.ascending_longitude.to(u.rad)}, w={state1_calc.periapsis_argument.to(u.rad)}')
    '''
    ukf = AdditiveUnscentedKalmanFilter()
    pp0 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    transition_cov = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    observation_cov = np.diag([0.1, 0.1, 0.1])
    def transition_fn(before_vec: np.ndarray) -> np.ndarray:
        before = MinorPlanetState.from_vector(before_vec, body, obs0.epoch)
        after = propagate(before, obs1.epoch)
        return after.to_vector()
    def observation_fn(state_vec: np.ndarray) -> np.ndarray:
        state = MinorPlanetState.from_vector(state_vec, body, obs1.epoch)
        measurement = measure_transform(state, obs1.observatory)
        return measurement.to_vector()
    x_next, pp_next = ukf.filter_update(state0_calc.to_vector(), pp0, obs1.to_vector(), transition_fn, transition_cov, observation_fn, observation_cov)
    pp_next = np.squeeze(pp_next)
    state1_filter = MinorPlanetState.from_vector(x_next, body, obs1.epoch)
    print(f'Filtered state k=1: pos={state1_filter.position.get_xyz().to(u.au)}, mm={state1_filter.mean_anomaly.to(u.rad)}, om={state1_filter.ascending_longitude.to(u.rad)}, w={state1_filter.periapsis_argument.to(u.rad)}')
    '''


    def transition_fn(before_vec: np.ndarray, dt: TimeDelta) -> np.ndarray:
        before = MinorPlanetState.from_vector(before_vec, body, obs0.epoch)
        after = propagate(before, delta_time=dt)
        return after.to_vector()


    def observation_fn(state_vec: np.ndarray) -> np.ndarray:
        state = MinorPlanetState.from_vector(state_vec, body, obs1.epoch)
        measurement = measure_transform(state, obs1.observatory)
        return measurement.to_vector()


    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=None, hx=observation_fn, fx=transition_fn, points=points)
    ukf.x = state0_calc.to_vector()
    ukf.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rr = np.diag([0.1, 0.1, 0.1])
    ukf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    dt0 = obs1.epoch - obs0.epoch
    ukf.predict(dt=dt0)
    ukf.update(z=obs1.to_vector(), R=rr)
    state1_filter = MinorPlanetState.from_vector(ukf.x, body, obs1.epoch)
    print(f'Filtered state k=1: pos={state1_filter.position.get_xyz().to(u.au)}, mm={state1_filter.mean_anomaly.to(u.rad)}, om={state1_filter.ascending_longitude.to(u.rad)}, w={state1_filter.periapsis_argument.to(u.rad)}')
