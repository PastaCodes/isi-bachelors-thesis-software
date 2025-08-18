import numpy as np
from astropy.coordinates import EarthLocation, CartesianRepresentation
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity

from orbit import get_mean_motion, compute_beta


class Observatory:
    def __init__(self, code: str, name: str, location: EarthLocation):
        self.code = code
        self.name = name
        self.location = location


class MinorPlanet:
    def __init__(self, name: str, hh: Quantity, gg: Quantity, a: Quantity, e: Quantity, i: Quantity,
                 observations_filepath: str | None = None, ephemeris_filepath: str | None = None,
                 initial_state_hint: 'StateHint | None' = None):
        self.name = name
        self.absolute_magnitude = hh
        self.slope = gg
        self.semi_major_axis = a
        self.eccentricity = e
        self.inclination = i
        self.observations_filepath = observations_filepath
        self.ephemeris_filepath = ephemeris_filepath
        self.initial_state_hint = initial_state_hint

        self.mean_motion = get_mean_motion(a)
        self.beta = compute_beta(e)
        self.minimum_distance = a * (1 - e)
        self.maximum_distance = a * (1 + e)


class MinorPlanetObservation:
    def __init__(self, target_body: MinorPlanet, epoch: Time, observatory: Observatory,
                 ra: Quantity, dec: Quantity, visual_mag: Quantity,
                 epoch_var: Quantity = 0, ra_var: Quantity = 0, dec_var: Quantity = 0, mag_var: Quantity = 0):
        self.target_body = target_body
        self.epoch = epoch
        self.epoch_variance = epoch_var
        self.observatory = observatory
        self.right_ascension = ra
        self.right_ascension_variance = ra_var
        self.declination = dec
        self.declination_variance = dec_var
        self.visual_magnitude = visual_mag
        self.magnitude_variance = mag_var

    def to_vector(self) -> np.ndarray:
        return np.array([self.right_ascension.to(u.rad).value,
                         self.declination.to(u.rad).value,
                         self.visual_magnitude.to(u.mag).value])

    @classmethod
    def from_vector(cls, measurement_vec: np.ndarray, target_body: MinorPlanet, epoch: Time, observatory: Observatory) -> 'MinorPlanetObservation':
        ra, dec, visual_mag = measurement_vec
        return MinorPlanetObservation(target_body=target_body, epoch=epoch, observatory=observatory,
                                      ra=(ra * u.rad), dec=(dec * u.rad), visual_mag=(visual_mag * u.mag))


class MinorPlanetState:
    def __init__(self, body: MinorPlanet, epoch: Time | None, pos: CartesianRepresentation, mm: Quantity,
                 asc_long: Quantity, peri_arg: Quantity):
        self.body = body
        self.epoch = epoch
        self.position = pos
        self.mean_anomaly = mm
        self.ascending_longitude = asc_long
        self.periapsis_argument = peri_arg

    def to_vector(self) -> np.ndarray:
        xyz = self.position.get_xyz()
        return np.array([xyz[0].to(u.au).value,
                         xyz[1].to(u.au).value,
                         xyz[2].to(u.au).value,
                         self.mean_anomaly.to(u.rad).value,
                         self.ascending_longitude.to(u.rad).value,
                         self.periapsis_argument.to(u.rad).value])

    @classmethod
    def from_vector(cls, vec: np.ndarray, body: MinorPlanet, epoch: Time | None) -> 'MinorPlanetState':
        x, y, z, mm, asc_long, peri_arg = vec
        return MinorPlanetState(body=body, epoch=epoch, pos=CartesianRepresentation(x, y, z, u.au), mm=(mm * u.rad),
                                asc_long=(asc_long * u.rad), peri_arg=(peri_arg * u.rad))


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
        self.elongation = th  # Not quite the true elongation
        self.phase = phi


class StateHint:
    def __init__(self, mean_anomaly: Quantity, ascending_longitude: Quantity, phase: Quantity = None):
        self.mean_anomaly = mean_anomaly
        self.ascending_longitude = ascending_longitude
        self.phase = phase
