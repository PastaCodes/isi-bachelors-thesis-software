import numpy.linalg as npl
from astropy.coordinates import get_body_barycentric, GCRS
from astropy.time import Time, TimeDelta

from classes import MinorPlanetObservation, MinorPlanetState, Observatory, StateHint
from orbit import *
from photometry import magnitude_from_state, distance_from_magnitude
from misc import law_of_cosines


def measure_transform(state: MinorPlanetState, observatory: Observatory) -> MinorPlanetObservation:
    # Earth w.r.t. SSB
    earth_pos: CartesianRepresentation = get_body_barycentric('earth', state.epoch, 'jpl')
    # Sun w.r.t. SSB
    sun_pos: CartesianRepresentation = get_body_barycentric('sun', state.epoch, 'jpl')

    # Observer w.r.t. Earth
    observer_geoc_pos, observer_geoc_vel = observatory.location.get_gcrs_posvel(state.epoch)
    # Target w.r.t. observer
    target_observer_pos = GCRS(state.position - earth_pos, obstime=state.epoch,
                               obsgeoloc=observer_geoc_pos, obsgeovel=observer_geoc_vel)

    observer_sun_dist = (observer_geoc_pos + earth_pos - sun_pos).norm()
    target_sun_dist = (state.position - sun_pos).norm()
    target_observer_dist = target_observer_pos.distance
    phase = law_of_cosines(target_observer_dist, target_sun_dist, observer_sun_dist)

    visual_mag = magnitude_from_state(state, phase, target_sun_dist, target_observer_dist)

    return MinorPlanetObservation(target_body=state.body,
                                  epoch=state.epoch,
                                  observatory=observatory,
                                  ra=target_observer_pos.ra,
                                  dec=target_observer_pos.dec,
                                  visual_mag=visual_mag)


def reverse_transform(obs: MinorPlanetObservation, hint: StateHint) -> MinorPlanetState:
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

    target_observer_dist = distance_from_magnitude(obs, elongation, sun_observer_dist)

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


def propagate(before_state: MinorPlanetState, after_time: Time | None = None,
              delta_time: TimeDelta | None = None) -> MinorPlanetState:
    body = before_state.body
    if delta_time is None:
        delta_time: TimeDelta = after_time - before_state.epoch
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
