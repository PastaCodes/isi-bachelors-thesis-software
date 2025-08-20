import numpy as np
from astropy.coordinates import EarthLocation, CartesianRepresentation
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity

from misc import arctan2pos_f
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
        ra = self.right_ascension.to(u.rad).value
        dec = self.declination.to(u.rad).value
        mag = self.visual_magnitude.to(u.mag).value
        return np.array([np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec), mag])

    @classmethod
    def from_vector(cls, measurement_vec: np.ndarray, target_body: MinorPlanet, epoch: Time,
                    observatory: Observatory) -> 'MinorPlanetObservation':
        cos_ra, sin_ra, cos_dec, sin_dec, visual_mag = measurement_vec
        ra = arctan2pos_f(sin_ra, cos_ra)
        dec = arctan2pos_f(sin_dec, cos_dec)
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
        x, y, z = self.position.get_xyz().to(u.au).value
        mm = self.mean_anomaly.to(u.rad).value
        asc_long = self.ascending_longitude.to(u.rad).value
        peri_arg = self.periapsis_argument.to(u.rad).value
        cos_mm = np.cos(mm)
        sin_mm = np.sin(mm)
        cos_asc_long = np.cos(asc_long)
        sin_asc_long = np.sin(asc_long)
        cos_peri_arg = np.cos(peri_arg)
        sin_peri_arg = np.sin(peri_arg)
        return np.array([x, y, z, cos_mm, sin_mm, cos_asc_long, sin_asc_long, cos_peri_arg, sin_peri_arg])

    @classmethod
    def from_vector(cls, vec: np.ndarray, body: MinorPlanet, epoch: Time | None) -> 'MinorPlanetState':
        x, y, z, cos_mm, sin_mm, cos_asc_long, sin_asc_long, cos_peri_arg, sin_peri_arg = vec
        mm = arctan2pos_f(sin_mm, cos_mm)
        asc_long = arctan2pos_f(cos_asc_long, sin_asc_long)
        peri_arg = arctan2pos_f(sin_peri_arg, cos_peri_arg)
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
