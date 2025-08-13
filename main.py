import numpy.linalg as npl
from astropy import units as u
from astropy.coordinates import GCRS, CartesianRepresentation, get_body_barycentric, get_body
from astropy.time import Time

from orbit import *
from misc import DecimalDayDatetime


class MinorPlanet:
    def __init__(self, name:str, observations_file: str, absolute_magnitude: float, slope: float,
                 # Begin with the dubious parameters
                 semi_major_axis: float,
                 eccentricity: float,
                 inclination: float) -> None:
        self.name = name
        self.observations_file = observations_file
        self.absolute_magnitude = absolute_magnitude
        self.slope = slope
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.inclination = inclination

        self.cot_i = 1 / math.tan(inclination)
        self.mean_motion = get_mean_motion(semi_major_axis)
        self.period = 2 * math.pi / self.mean_motion
        self.beta = eccentricity / (1 + math.sqrt(1 - eccentricity ** 2))

    def load_observations(self) -> list['MinorPlanetObservation']:
        from parse import parse_obs_file
        return parse_obs_file(self, accept_methods=['B', 'C'])


# As described by https://www.minorplanetcenter.net/iau/info/OpticalObs.html
class MinorPlanetObservation:
    def __init__(self, obj: MinorPlanet, datetime: DecimalDayDatetime, year_sd: float,
                 right_ascension: float, right_ascension_sd: float,
                 declination: float, declination_sd: float, magnitude: float, magnitude_sd: float, band: str,
                 is_discovery: bool) -> None:
        self.obj = obj
        self.datetime = datetime
        self.year_sd = year_sd
        self.right_ascension = right_ascension
        self.right_ascension_sd = right_ascension_sd
        self.declination = declination
        self.declination_sd = declination_sd
        self.magnitude = magnitude
        self.magnitude_sd = magnitude_sd
        self.band = band
        self.is_discovery = is_discovery

        self.obstime = Time(datetime.to_datetime(), scale='utc')
        from photometry import visual_magnitude_from_observed
        self.visual_magnitude = visual_magnitude_from_observed(magnitude, band)

    def to_angle_coord(self) -> GCRS:
        return GCRS(ra=(self.right_ascension * u.hourangle), dec=(self.declination * u.degree), obstime=self.obstime)

    def to_absolute_coord(self, distance: float) -> CartesianRepresentation:
        relative = GCRS(ra=(self.right_ascension * u.hourangle), dec=(self.declination * u.degree),
                        distance=(distance * u.au), obstime=self.obstime)
        earth_position: CartesianRepresentation = get_body_barycentric('earth', self.obstime, 'jpl')
        return relative.cartesian + earth_position


class MinorPlanetState:
    def __init__(self, obj: MinorPlanet, time: Time, position: np.ndarray, distance: float, mean_anomaly: float,
                 orbit_plane_angle: float, z_angle_offset: float):
        self.obj = obj
        self.time = time
        self.position = position
        self.distance = distance
        self.mean_anomaly = mean_anomaly
        self.orbit_plane_angle = orbit_plane_angle
        self.z_angle_offset = z_angle_offset

    @classmethod
    def from_position(cls, obj: MinorPlanet, time: Time, position: np.ndarray,
                      state_hint: 'MinorPlanetState | MinorPlanetEphemeris') -> 'MinorPlanetState':
        distance = npl.norm(position)
        return cls(obj, time, position, distance, None, None, None)
        eccentric_anomaly = eccentric_anomaly_from_distance(distance, obj.semi_major_axis, obj.eccentricity,
                                                            state_hint.mean_anomaly)
        mean_anomaly = mean_anomaly_from_eccentric_anomaly(eccentric_anomaly, obj.eccentricity)
        true_anomaly = true_anomaly_from_eccentric_anomaly(eccentric_anomaly, obj.beta)
        z_angle_offset, orbit_plane_angle = z_angles(position, true_anomaly, obj.cot_i)
        return cls(obj, time, position, distance, mean_anomaly, orbit_plane_angle, z_angle_offset)

    @classmethod
    def from_anomaly(cls, obj: MinorPlanet, time: Time, mean_anomaly: float, orbit_plane_angle: float,
                     z_angle_offset: float) -> 'MinorPlanetState':
        eccentric_anomaly = eccentric_anomaly_from_mean_anomaly(mean_anomaly, obj.eccentricity)
        distance = distance_from_eccentric_anomaly(eccentric_anomaly, obj.semi_major_axis, obj.eccentricity)
        true_anomaly = true_anomaly_from_eccentric_anomaly(eccentric_anomaly, obj.beta)
        position = position_from_orbit_angles(true_anomaly, z_angle_offset, orbit_plane_angle,
                                              obj.inclination, distance)
        return cls(obj, time, position, distance, mean_anomaly, orbit_plane_angle, z_angle_offset)


class MinorPlanetEphemeris:
    def __init__(self, obj: MinorPlanet, position: np.ndarray, semi_major_axis: float, eccentricity: float,
                 inclination: float, mean_motion: float, mean_anomaly: float, true_anomaly: float,
                 right_ascension: float, declination: float, apparent_ra: float, apparent_decl: float,
                 visual_magnitude: float, elongation: float, phase: float) -> None:
        self.obj = obj
        self.position = position
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.mean_motion = mean_motion
        self.mean_anomaly = mean_anomaly
        self.true_anomaly = true_anomaly
        self.right_ascension = right_ascension
        self.declination = declination
        self.apparent_ra = apparent_ra
        self.apparent_decl = apparent_decl
        self.visual_magnitude = visual_magnitude
        self.elongation = elongation
        self.phase = phase


def elongation_from_observation(obs: MinorPlanetObservation) -> float:
    obj = obs.to_angle_coord()
    sun: GCRS = get_body('sun', obs.obstime, None, 'jpl')
    return obj.separation(sun).to(u.rad).value


def earth_sun_distance(time: Time) -> float:
    earth_position = get_body_barycentric('earth', time, 'jpl')
    sun_position: CartesianRepresentation = get_body_barycentric('sun', time, 'jpl')
    return (earth_position - sun_position).norm().to(u.au).value


def output_transform(obs: MinorPlanetObservation,
                     state_hint: MinorPlanetState | MinorPlanetEphemeris) -> MinorPlanetState:
    from photometry import distance_from_magnitude
    distance = distance_from_magnitude(obs)
    position: np.ndarray = obs.to_absolute_coord(distance).get_xyz().to(u.au).value
    return MinorPlanetState.from_position(obs.obj, obs.obstime, position, state_hint)


def propagate(before_time: float, after_time: float, before_state: MinorPlanetState) -> MinorPlanetState:
    obj = before_state.obj
    delta_time = after_time - before_time
    return MinorPlanetState.from_anomaly(obj=obj, time=before_state.time,
                                         mean_anomaly=advance_mean_anomaly(before=before_state.mean_anomaly,
                                                                           mean_motion=obj.mean_motion,
                                                                           delta_time=delta_time),
                                         orbit_plane_angle=before_state.orbit_plane_angle,
                                         z_angle_offset=before_state.z_angle_offset)


def track(obj: MinorPlanet) -> None:
    obss = obj.load_observations()
    state0 = output_transform(obss[0])
    before_state = state0
    i = 1
    while i < len(obss):
        obs = obss[i]
        if obs.obstime == before_state.time:
            i += 1
            continue
        predicted_state = propagate(before_time=obs.datetime.to_decimal_year(),
                                    after_time=obs.datetime.to_decimal_year(),
                                    before_state=before_state)
        observed_state = output_transform(obs, predicted_state)
        # input('Press Enter to continue...')
        before_state = observed_state
        i += 1


#                            Name         File                 H      G     a      e      i
OBJ_BENNU     = MinorPlanet('Bennu',     'data/Bennu.txt',     20.2, -0.03, 1.128, 0.204, 0.514)
OBJ_2024_YR4  = MinorPlanet('2024 YR4',  'data/2024_yr4.txt',  23.9,  0.15, 2.516, 0.662, 0.469)
OBJ_2010_RF12 = MinorPlanet('2010 RF12', 'data/2010_rf12.txt', 28.5,  0.15, 1.061, 0.188, 0.424)


if __name__ == '__main__':
    obss = OBJ_BENNU.load_observations()
    print(obss[130].datetime.to_datetime())
