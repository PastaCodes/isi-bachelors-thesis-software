from functools import cached_property

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from misc import angle_components, angle_from_components
from orbit import get_mean_motion
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
                 observations_filepath: str | None = None, ephemeris_filepath: str | None = None,
                 a0: float = 1.0, e0: float = 0.5, i0: float = 0.25 * np.pi, n0: float | None = None,
                 hh0: float = 20.0, gg0: float = 0.15, mm0_hint: float = 0.0, om0_hint: float = 0.0,
                 mm0_var: float = 1E-3, a0_var: float = 1E-5, e0_var: float = 1E-5, i0_var: float = 1E-5,
                 om0_var: float = 1E-3, w0_var: float = 1E-3, n0_var: float = 1E-3,
                 hh0_var: float = 1E-5, gg0_var: float = 1E-3,
                 mm_var: float = 1E-5, a_var: float = 1E-7, e_var: float = 1E-7, i_var: float = 1E-7,
                 om_var: float = 1E-7, w_var: float = 1E-7, n_var: float = 1E-7,
                 hh_var: float = 1E-12, gg_var: float = 1E-12, dir_var: float = 1E-5, vv_var: float = 1E-2,
                 dt_exp: float = 1.0):
        self.display_name = jpl_designation if display_name is None else display_name
        self.jpl_designation = display_name if jpl_designation is None else jpl_designation
        if isinstance(additional_designations, set):
            self.additional_designations = additional_designations
        elif isinstance(additional_designations, str):
            self.additional_designations = {additional_designations}
        else:
            self.additional_designations = set()
        self.a0 = a0
        self.e0 = e0
        self.i0 = i0
        self.n0 = get_mean_motion(a0) if n0 is None else n0
        self.hh0 = hh0
        self.gg0 = gg0
        self.mm0_hint = mm0_hint
        self.om0_hint = om0_hint
        self.mm0_var = mm0_var
        self.a0_var = a0_var
        self.e0_var = e0_var
        self.i0_var = i0_var
        self.om0_var = om0_var
        self.w0_var = w0_var
        self.n0_var = n0_var
        self.hh0_var = hh0_var
        self.gg0_var = gg0_var
        self.mm_var = mm_var
        self.a_var = a_var
        self.e_var = e_var
        self.i_var = i_var
        self.om_var = om_var
        self.w_var = w_var
        self.n_var = n_var
        self.hh_var = hh_var
        self.gg_var = gg_var
        self.dir_var = dir_var
        self.vv_var = vv_var
        self.dt_exp = dt_exp
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


class MinorPlanetEstimate:
    def __init__(self, body: MinorPlanet, epoch: Time, pos: np.ndarray, mm: float, a: float, e: float, i: float,
                 om: float, w: float, n: float, hh: float, gg: float):
        self.body = body
        self.epoch = epoch
        self.position = pos
        self.mean_anomaly = mm
        self.semi_major_axis = a
        self.eccentricity = e
        self.inclination = i
        self.ascending_longitude = om
        self.periapsis_argument = w
        self.mean_motion = n
        self.absolute_magnitude = hh
        self.slope_parameter = gg

    @classmethod
    def from_state_vector(cls, state_vec: np.ndarray, body: MinorPlanet, epoch: Time) -> 'MinorPlanetEstimate':
        from model import finalize_transform
        pos = finalize_transform(state_vec)
        cos_mm, sin_mm, a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg = state_vec
        return cls(body, epoch, pos, angle_from_components(cos_mm, sin_mm), a, e, angle_from_components(cos_i, sin_i),
                   angle_from_components(cos_om, sin_om), angle_from_components(cos_w, sin_w), n, hh, gg)


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
        return visual_magnitude_from_absolute(self.target_body.hh0, self.geometric_target_sun_distance,
                                              self.geometric_target_observer_distance, self.geometric_phase,
                                              self.target_body.gg0)

    def to_state_vector(self) -> np.ndarray:
        return np.array([*angle_components(self.mean_anomaly), self.semi_major_axis, self.eccentricity,
                         *angle_components(self.inclination), *angle_components(self.ascending_longitude),
                         *angle_components(self.periapsis_argument), self.mean_motion,
                         self.target_body.hh0, self.target_body.gg0])

    def to_geometric_measurement_vector(self) -> np.ndarray:
        return np.array([*angle_components(self.geometric_right_ascension),
                         *angle_components(self.geometric_declination), self.geometric_visual_magnitude])
