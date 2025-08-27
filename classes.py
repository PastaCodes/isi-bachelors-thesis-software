from functools import cached_property

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
    def __init__(self, display_name: str | None = None, jpl_designation: str | None = None,
                 additional_designations: str | set[str] | None = None,
                 observations_filepath: str | None = None, ephemeris_filepath: str | None = None):
        self.display_name = jpl_designation if display_name is None else display_name
        self.jpl_designation = display_name if jpl_designation is None else jpl_designation
        if isinstance(additional_designations, set):
            self.additional_designations = additional_designations
        elif isinstance(additional_designations, str):
            self.additional_designations = {additional_designations}
        else:
            self.additional_designations = set()
        self.observations_filepath = observations_filepath
        self.ephemeris_filepath = ephemeris_filepath

    @cached_property
    def all_designations(self):
        all_designations = set()
        if self.display_name:
            all_designations.add(self.display_name)
        if self.jpl_designation:
            all_designations.add(self.jpl_designation)
        all_designations.update(self.additional_designations)
        return all_designations


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


class MinorPlanetEphemeris:
    def __init__(self, target_body: MinorPlanet, epoch: Time, target_pos: np.ndarray,
                 observer_pos: np.ndarray, sun_pos: np.ndarray,
                 a: float, e: float, i: float, n: float, om: float, w: float,
                 mm: float, v: float, astro_ra: float, astro_dec: float, app_ra: float, app_dec: float,
                 app_vv: float, app_th: float, app_phi: float, astro_phi: float):
        self.target_body = target_body
        self.epoch = epoch
        self.target_position = target_pos
        self.observer_position = observer_pos
        self.sun_position = sun_pos
        self.semi_major_axis = a
        self.eccentricity = e
        self.inclination = i
        self.mean_motion = n
        self.ascending_longitude = om
        self.periapsis_argument = w
        self.mean_anomaly = mm
        self.true_anomaly = v
        self.astrometric_right_ascension = astro_ra
        self.astrometric_declination = astro_dec
        self.apparent_right_ascension = app_ra
        self.apparent_declination = app_dec
        self.apparent_visual_magnitude = app_vv
        self.apparent_elongation = app_th
        self.apparent_phase = app_phi
        self.astrometric_phase = astro_phi
