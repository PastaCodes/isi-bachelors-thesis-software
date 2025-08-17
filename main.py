import numpy.linalg as npl
from astropy import units as u
from astropy.coordinates import GCRS, get_body_barycentric, EarthLocation
from astropy.time import Time, TimeDelta

from orbit import *


class Observatory:
    def __init__(self, code: str, name: str, location: EarthLocation):
        self.code = code
        self.name = name
        self.location = location


class MinorPlanet:
    def __init__(self, name: str, hh: Quantity, gg: Quantity, a: Quantity, e: Quantity, i: Quantity,
                 observations_filepath: str | None = None, ephemeris_filepath: str | None = None):
        self.name = name
        self.absolute_magnitude = hh
        self.slope = gg
        self.semi_major_axis = a
        self.eccentricity = e
        self.inclination = i
        self.observations_filepath = observations_filepath
        self.ephemeris_filepath = ephemeris_filepath

        self.mean_motion = get_mean_motion(a)
        self.beta = compute_beta(e)
        self.minimum_distance = a * (1 - e)
        self.maximum_distance = a * (1 + e)


class MinorPlanetObservation:
    def __init__(self, body: MinorPlanet, epoch: Time, observatory: Observatory, ra: Quantity, dec: Quantity,
                 mag: Quantity, band: str, epoch_var: Quantity = 0, ra_var: Quantity = 0, dec_var: Quantity = 0,
                 mag_var: Quantity = 0):
        self.target_body = body
        self.epoch = epoch
        self.epoch_variance = epoch_var
        self.observatory = observatory
        self.right_ascension = ra
        self.right_ascension_variance = ra_var
        self.declination = dec
        self.declination_variance = dec_var
        self.magnitude = mag
        self.magnitude_variance = mag_var
        self.band = band


class MinorPlanetState:
    def __init__(self, body: MinorPlanet, epoch: Time, pos: CartesianRepresentation, mm: Quantity,
                 asc_long: Quantity, peri_arg: Quantity):
        self.body = body
        self.epoch = epoch
        self.position = pos
        self.mean_anomaly = mm
        self.ascending_longitude = asc_long
        self.periapsis_argument = peri_arg


class MinorPlanetEphemeris:
    def __init__(self, body: MinorPlanet, epoch: Time, observer_loc: EarthLocation, pos: CartesianRepresentation,
                 a: Quantity, e: Quantity, i: Quantity, n: Quantity, asc_long: Quantity, peri_arg: Quantity,
                 mm: Quantity, v: Quantity, ra: Quantity, dec: Quantity, app_ra: Quantity, app_dec: Quantity,
                 vv: Quantity, th: Quantity, phi: Quantity):
        self.body = body
        self.epoch = epoch
        self.observer_location = observer_loc
        self.position = pos
        self.semi_major_axis = a
        self.eccentricity = e
        self.inclination = i
        self.mean_motion = n
        self.ascending_longitude = asc_long
        self.periapsis_argument = peri_arg
        self.mean_anomaly = mm
        self.true_anomaly = v
        self.right_ascension = ra
        self.declination = dec
        self.apparent_right_ascension = app_ra
        self.apparent_declination = app_dec
        self.visual_magnitude = vv
        self.elongation = th  # Apparent elongation
        self.phase = phi


def output_transform(obs: MinorPlanetObservation,
                     hint: MinorPlanetState | MinorPlanetEphemeris) -> MinorPlanetState:
    body = obs.target_body

    # Earth w.r.t. SSB
    earth_pos: CartesianRepresentation = get_body_barycentric('earth', obs.epoch, 'jpl')
    # Sun w.r.t. SSB
    sun_pos: CartesianRepresentation = get_body_barycentric('sun', obs.epoch, 'jpl')

    # Observer w.r.t. Earth
    observer_geoc_pos, observer_geoc_vel = obs.observatory.location.get_gcrs_posvel(obs.epoch)
    # Observer w.r.t. SSB
    observer_pos: CartesianRepresentation = observer_geoc_pos + earth_pos

    # Sun w.r.t. observer
    sun_observer_pos = GCRS(sun_pos - earth_pos, obstime=obs.epoch,
                            obsgeoloc=observer_geoc_pos, obsgeovel=observer_geoc_vel)
    # Target w.r.t. observer (WITHOUT information on the distance)
    target_observer_pos = GCRS(ra=obs.right_ascension, dec=obs.declination, obstime=obs.epoch,
                               obsgeoloc=observer_geoc_pos, obsgeovel=observer_geoc_vel)

    elongation = target_observer_pos.separation(sun_observer_pos)

    sun_observer_dist: Quantity = sun_observer_pos.distance

    from photometry import visual_magnitude_from_observed, distance_from_magnitude
    visual_magnitude = visual_magnitude_from_observed(obs.magnitude, obs.band)
    target_observer_dist = distance_from_magnitude(obs, visual_magnitude, elongation, sun_observer_dist)

    # Target w.r.t. observer (now WITH information on the distance)
    target_observer_pos = GCRS(ra=obs.right_ascension, dec=obs.declination, distance=target_observer_dist,
                               obstime=obs.epoch, obsgeoloc=observer_geoc_pos, obsgeovel=observer_geoc_vel)
    # Target w.r.t. SSB
    target_pos = target_observer_pos.cartesian + observer_pos

    orbit_dist = npl.norm(target_pos.get_xyz())
    eccentric_anomaly = eccentric_anomaly_from_distance(orbit_dist, body.semi_major_axis, body.eccentricity,
                                                        hint.mean_anomaly)
    mean_anomaly = mean_anomaly_from_eccentric_anomaly(eccentric_anomaly, body.eccentricity)
    true_anomaly = true_anomaly_from_eccentric_anomaly(eccentric_anomaly, body.beta)

    asc_long, peri_arg = orbital_angles_from_position(target_pos, true_anomaly, body.inclination,
                                                      hint.ascending_longitude)

    return MinorPlanetState(body, obs.epoch, target_pos, mean_anomaly, asc_long, peri_arg)


def propagate(before_time: Time, after_time: Time, before_state: MinorPlanetState) -> MinorPlanetState:
    body = before_state.body
    delta_time: TimeDelta = after_time - before_time
    mean_anomaly = advance_mean_anomaly(before=before_state.mean_anomaly,
                                        mean_motion=body.mean_motion,
                                        delta_time=delta_time.to(u.year))
    eccentric_anomaly = eccentric_anomaly_from_mean_anomaly(mean_anomaly, body.eccentricity)
    distance = distance_from_eccentric_anomaly(eccentric_anomaly, body.semi_major_axis, body.eccentricity)
    true_anomaly = true_anomaly_from_eccentric_anomaly(eccentric_anomaly, body.beta)
    position = position_from_orbital_angles(true_anomaly, before_state.periapsis_argument,
                                            before_state.ascending_longitude, body.inclination, distance)
    return MinorPlanetState(body, after_time, position, mean_anomaly,
                            before_state.ascending_longitude, before_state.periapsis_argument)


def track(obj: MinorPlanet) -> None:
    obss = obj.load_observations()
    state0 = output_transform(obss[0])
    before_state = state0
    i = 1
    while i < len(obss):
        obs = obss[i]
        if obs.time == before_state.time:
            i += 1
            continue
        predicted_state = propagate(before_time=obs.time.to_value('decimalyear'),
                                    after_time=obs.time.to_value('decimalyear'),
                                    before_state=before_state)
        observed_state = output_transform(obs, predicted_state)
        # input('Press Enter to continue...')
        before_state = observed_state
        i += 1


BODY_BENNU = MinorPlanet(name='1999 RQ36',
                         hh=(20.21 * u.mag),
                         gg=(-0.031 * u.dimensionless_unscaled),
                         a=(1.128 * u.au),
                         e=(0.204 * u.dimensionless_unscaled),
                         i=(0.514 * u.rad),
                         observations_filepath='data/Bennu.txt',
                         ephemeris_filepath='data/Bennu_eph.txt')

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
    from query import query_ephemeris
    ephs = query_ephemeris(BODY_2024_YR4)
    from parse import dump_ephemeris
    dump_ephemeris(ephs, BODY_2024_YR4.ephemeris_filepath)
