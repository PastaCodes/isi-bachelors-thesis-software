import numpy as np
from astropy.coordinates import get_body_barycentric, EarthLocation, CartesianRepresentation
from astropy.time import Time
import astropy.units as u

from misc import safe_arcsin, arctan2pos, law_of_cosines, unit_vector_separation


def elongation_from_distances(rr: float, delta: float, d: float) -> float:
    return law_of_cosines(rr, delta, d)


def elongation_from_directions(tgt_dir: np.ndarray, sun_dir: np.ndarray) -> float:
    return unit_vector_separation(tgt_dir, sun_dir)


def phase_from_distances(d: float, delta: float, rr: float) -> float:
    return law_of_cosines(d, delta, rr)


def direction_to_ra_dec(dir_vec: np.ndarray) -> tuple[float, float]:
    # noinspection PyTypeChecker
    ra = arctan2pos(dir_vec[1], dir_vec[0])
    # noinspection PyTypeChecker
    dec = safe_arcsin(dir_vec[2])
    return ra, dec


def direction_to_ra_dec_components(dir_vec: np.ndarray) -> tuple[float, float, float, float]:
    sin_dec = dir_vec[2]
    cos_dec = np.sqrt(1.0 - sin_dec * sin_dec)
    if cos_dec == 0.0:
        cos_ra = 1.0
        sin_ra = 0.0
    else:
        cos_ra = dir_vec[0] / cos_dec
        sin_ra = dir_vec[1] / cos_dec
    # noinspection PyTypeChecker
    return cos_ra, sin_ra, cos_dec, sin_dec


def ra_dec_components_to_direction(cos_ra: float, sin_ra: float, cos_dec: float, sin_dec: float) -> np.ndarray:
    return np.array([cos_ra * cos_dec, sin_ra * cos_dec, sin_dec])


def get_sun_position(epoch: Time) -> np.ndarray:
    return get_body_barycentric('sun', epoch, 'jpl').get_xyz().to_value(u.au)


def get_location_position(location: EarthLocation, epoch: Time) -> np.ndarray:
    earth_pos: CartesianRepresentation = get_body_barycentric('earth', epoch, 'jpl')
    observer_geocentric_pos, _ = location.get_gcrs_posvel(epoch)
    return (observer_geocentric_pos + earth_pos).get_xyz().to_value(u.au)
