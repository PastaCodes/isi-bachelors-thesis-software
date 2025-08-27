import numpy as np

from misc import unit_vector_separation, angle_components, law_of_cosines, norm, angle_from_components
from orbit import (orbital_angles_from_position, true_anomaly_from_eccentric_anomaly, eccentric_anomaly_from_distance,
                   mean_anomaly_from_eccentric_anomaly, eccentric_anomaly_from_mean_anomaly,
                   distance_from_eccentric_anomaly, position_from_orbital_angle_components, advance_mean_anomaly)
from photometry import solve_distance_and_phase, visual_magnitude_from_absolute


def reverse_transform(obs_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
                      hh: float, gg: float) -> np.ndarray:
    """
    [*α, *δ, V] --> [x, y, z]
    """
    cos_ra, sin_ra, cos_dec, sin_dec, vv = obs_vec
    tgt_obs_unit_vec = np.array([cos_ra * cos_dec, sin_ra * cos_dec, sin_dec])
    sun_obs_pos = sun_pos - obs_pos
    sun_obs_dist = norm(sun_obs_pos)
    sun_obs_unit_vec = sun_obs_pos / sun_obs_dist
    th = unit_vector_separation(tgt_obs_unit_vec, sun_obs_unit_vec)
    tgt_obs_dist, _ = solve_distance_and_phase(vv, sun_obs_dist, th, hh, gg)
    tgt_obs_pos = tgt_obs_dist * tgt_obs_unit_vec
    tgt_pos = tgt_obs_pos + obs_pos
    return tgt_pos


def initial_state(obs_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
                  a: float, e: float, i: float, n: float, hh: float, gg: float,
                  mm_hint: float, om_hint: float) -> np.ndarray:
    tgt_pos = reverse_transform(obs_vec, obs_pos, sun_pos, hh, gg)
    r = norm(tgt_pos)
    ee = eccentric_anomaly_from_distance(r, a, e, mm_hint)
    mm = mean_anomaly_from_eccentric_anomaly(ee, e)
    v = true_anomaly_from_eccentric_anomaly(ee, e)
    om, w = orbital_angles_from_position(tgt_pos, v, i, om_hint)
    return np.array([*angle_components(mm), a, e, *angle_components(i),
                     *angle_components(om), *angle_components(w), n, hh, gg])


def finalize_transform(state_vec: np.ndarray) -> np.ndarray:
    """
    [*M, a, e, *i, *Ω, *ω, n, H, G] --> [x, y, z]
    """
    cos_mm, sin_mm, a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg = state_vec
    mm = angle_from_components(cos_mm, sin_mm)
    ee = eccentric_anomaly_from_mean_anomaly(mm, e)
    v = true_anomaly_from_eccentric_anomaly(ee, e)
    cos_v, sin_v = angle_components(v)
    r = distance_from_eccentric_anomaly(ee, a, e)
    return position_from_orbital_angle_components(cos_om, sin_om, cos_w, sin_w, cos_v, sin_v, cos_i, sin_i, r)


def measure(state_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray) -> np.ndarray:
    """
    [*M, a, e, *i, *Ω, *ω, n, H, G]_k --> [*α, *δ, V]_k
    """
    tgt_pos = finalize_transform(state_vec)
    hh, gg = state_vec[-2:]
    tgt_obs_pos = tgt_pos - obs_pos
    tgt_obs_dist = norm(tgt_obs_pos)
    tgt_sun_dist = norm(tgt_pos - sun_pos)
    obs_sun_dist = norm(obs_pos - sun_pos)
    sin_dec = tgt_obs_pos[2] / tgt_obs_dist
    cos_dec = np.sqrt(1.0 - sin_dec * sin_dec)
    cos_ra = tgt_obs_pos[0] / (tgt_obs_dist * cos_dec)
    sin_ra = tgt_obs_pos[1] / (tgt_obs_dist * cos_dec)
    phi = law_of_cosines(tgt_obs_dist, tgt_sun_dist, obs_sun_dist)
    vv = visual_magnitude_from_absolute(hh, tgt_sun_dist, tgt_obs_dist, phi, gg)
    return np.array([cos_ra, sin_ra, cos_dec, sin_dec, vv])


def propagate(before_vec: np.ndarray, dt: float) -> np.ndarray:
    """
    [*M, a, e, *i, *Ω, *ω, n, H, G]_k-1 --> [*M, a, e, *i, *Ω, *ω, n, H, G]_k
    """
    cos_mm_before, sin_mm_before, a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg = before_vec
    mm_before = angle_from_components(cos_mm_before, sin_mm_before)
    mm_after = advance_mean_anomaly(mm_before, n, dt)
    return np.array([*angle_components(mm_after), a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg])
