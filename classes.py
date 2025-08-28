from functools import cached_property

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from misc import angle_components
from photometry import visual_magnitude_from_absolute
from sky import direction_to_ra_dec, elongation_from_distances, phase_from_distances


class Observatory:
    def __init__(self, code: str, name: str, location: EarthLocation):
        self.code = code
        self.name = name
        self.location = location


class MinorPlanet:
    def __init__(self, display_name: str | None = None, jpl_designation: str | None = None,
                 additional_designations: str | set[str] | None = None,
                 absolute_magnitude: float | None = None, slope_parameter: float | None = 0.15,
                 observations_filepath: str | None = None, ephemeris_filepath: str | None = None):
        self.display_name = jpl_designation if display_name is None else display_name
        self.jpl_designation = display_name if jpl_designation is None else jpl_designation
        if isinstance(additional_designations, set):
            self.additional_designations = additional_designations
        elif isinstance(additional_designations, str):
            self.additional_designations = {additional_designations}
        else:
            self.additional_designations = set()
        self.absolute_magnitude = absolute_magnitude
        self.slope_parameter = slope_parameter
        self.observations_filepath = observations_filepath
        self.ephemeris_filepath = ephemeris_filepath

    @cached_property
    def all_designations(self) -> set[str]:
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


class MinorPlanetEphemeris:
    def __init__(self, target_body: MinorPlanet, epoch: Time, target_pos: np.ndarray,
                 observer_pos: np.ndarray, sun_pos: np.ndarray,
                 a: float, e: float, i: float, om: float, w: float, mm: float, v: float, n: float,
                 astro_ra: float, astro_dec: float, app_ra: float, app_dec: float,
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

    @cached_property
    def target_topocentric_position(self) -> np.ndarray:
        return self.target_position - self.observer_position

    @cached_property
    def geometric_target_observer_distance(self) -> float:
        # noinspection PyTypeChecker
        return np.linalg.norm(self.target_topocentric_position)

    @cached_property
    def geometric_target_sun_distance(self) -> float:
        # noinspection PyTypeChecker
        return np.linalg.norm(self.target_position - self.sun_position)

    @cached_property
    def geometric_observer_sun_distance(self) -> float:
        # noinspection PyTypeChecker
        return np.linalg.norm(self.observer_position - self.sun_position)

    @cached_property
    def geometric_elongation(self) -> float:
        return elongation_from_distances(self.geometric_observer_sun_distance, self.geometric_target_observer_distance,
                                         self.geometric_target_sun_distance)

    @cached_property
    def geometric_phase(self) -> float:
        return phase_from_distances(self.geometric_target_sun_distance, self.geometric_target_observer_distance,
                                    self.geometric_observer_sun_distance)

    @cached_property
    def observation_direction(self) -> np.ndarray:
        return self.target_topocentric_position / self.geometric_target_observer_distance

    @cached_property
    def _geom_ra_dec(self) -> tuple[float, float]:
        return direction_to_ra_dec(self.observation_direction)

    @cached_property
    def geometric_right_ascension(self) -> float:
        return self._geom_ra_dec[0]

    @cached_property
    def geometric_declination(self) -> float:
        return self._geom_ra_dec[1]

    @cached_property
    def geometric_visual_magnitude(self) -> float:
        return visual_magnitude_from_absolute(self.target_body.absolute_magnitude, self.geometric_target_sun_distance,
                                              self.geometric_target_observer_distance, self.geometric_phase,
                                              self.target_body.slope_parameter)

    def to_state_vector(self) -> np.ndarray:
        return np.array([*angle_components(self.mean_anomaly), self.semi_major_axis, self.eccentricity,
                         *angle_components(self.inclination), *angle_components(self.ascending_longitude),
                         *angle_components(self.periapsis_argument), self.mean_motion,
                         self.target_body.absolute_magnitude, self.target_body.slope_parameter])

    def to_geometric_measurement_vector(self) -> np.ndarray:
        return np.array([*angle_components(self.geometric_right_ascension),
                         *angle_components(self.geometric_declination), self.geometric_visual_magnitude])
