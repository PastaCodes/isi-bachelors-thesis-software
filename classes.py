from typing import Collection

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from misc import angle_components


class Observatory:
    def __init__(self, code: str, name: str, location: EarthLocation):
        self.code = code
        self.name = name
        self.location = location


class MinorPlanet:
    def __init__(self, designations: str | Collection[str], observations_filepath: str | None = None,
                 ephemeris_filepath: str | None = None):
        self.designations = designations if isinstance(designations, list) else [designations]
        self.observations_filepath = observations_filepath
        self.ephemeris_filepath = ephemeris_filepath


class MinorPlanetObservation:
    def __init__(self, target_body: MinorPlanet, epoch: Time, observatory: Observatory,
                 ra: float, dec: float, vv: float):
        self.target_body = target_body
        self.epoch = epoch
        self.observatory = observatory
        self.right_ascension = ra
        self.declination = dec
        self.visual_magnitude = vv

    @classmethod
    def with_band(cls, target_body: MinorPlanet, epoch: Time, observatory: Observatory,
                  ra: float, dec: float, observed_magnitude: float, band: str):
        from photometry import visual_magnitude_from_observed
        vv = visual_magnitude_from_observed(observed_magnitude, band)
        return cls(target_body, epoch, observatory, ra, dec, vv)

    def to_vector(self) -> np.ndarray:
        return np.array([*angle_components(self.right_ascension),
                         *angle_components(self.declination),
                         self.visual_magnitude])


class MinorPlanetState:
    def __init__(self, body: MinorPlanet, epoch: Time, mm: float, a: float, e: float, i: float, om: float, w: float,
                 n: float, hh: float, gg: float):
        self.body = body
        self.epoch = epoch
        self.mean_anomaly = mm
        self.semi_major_axis = a
        self.eccentricity = e
        self.inclination = i
        self.ascending_longitude = om
        self.periapsis_argument = w
        self.mean_motion = n
        self.absolute_magnitude = hh
        self.slope_parameter = gg


'''
class MinorPlanetEphemeris:
    def __init__(self, target_body: MinorPlanet, epoch: Time, target_pos: CartesianRepresentation,
                 observer_loc: EarthLocation, observer_pos: CartesianRepresentation, sun_pos: CartesianRepresentation,
                 a: Quantity, e: Quantity, i: Quantity, n: Quantity, asc_long: Quantity, peri_arg: Quantity,
                 mm: Quantity, v: Quantity, ra: Quantity, dec: Quantity, app_ra: Quantity, app_dec: Quantity,
                 vv: Quantity, th: Quantity, phi: Quantity):
        """
        :param target_body:
        :param epoch:
        :param target_pos: Geometric position of the target's barycenter
        :param observer_loc: Earth location of the observer
        :param observer_pos: Geometric position of the observer
        :param sun_pos: Geometric position of the Sun's barycenter
        :param a:
        :param e:
        :param i:
        :param n:
        :param asc_long:
        :param peri_arg:
        :param mm:
        :param v:
        :param ra: Astrometric right ascension
        :param dec: Astrometric declination
        :param app_ra: Refracted apparent right ascension
        :param app_dec: Refracted apparent declination
        :param vv:
        :param th:
        :param phi: True phase
        """
        self.target_body = target_body
        self.epoch = epoch
        self.target_position = target_pos
        self.observer_location = observer_loc
        self.observer_position = observer_pos
        self.sun_position = sun_pos
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
'''
