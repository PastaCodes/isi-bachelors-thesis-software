from typing import Final, Collection

import numpy as np
from fortranformat import FortranRecordReader

from classes import MinorPlanetObservation, MinorPlanet, Observatory
from misc import location_from_parallax, decimal_day_date_to_time, PROJECT_ROOT


# As described in https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html
OBSERVATORY_FORMAT: Final[str] = '(A3, 1X, F9.5, F8.6, F9.6, A50)'
observatory_reader = FortranRecordReader(OBSERVATORY_FORMAT)
observatories_file_path = 'data/ObsCodes.txt'
observatories: dict[str, Observatory] = {}


def load_observatories():
    with open(PROJECT_ROOT + observatories_file_path, 'rt', encoding='utf-8') as file:
        for line in file.readlines()[1:]:
            if not line[6]:  # No location data; likely not on Earth
                continue
            code, lon_deg, p1, p2, name = observatory_reader.read(line)
            loc = location_from_parallax(float(lon_deg), float(p1), float(p2))
            observatories[code] = Observatory(code, name, loc)


# As described in https://www.minorplanetcenter.net/iau/info/OpticalObs.html
OBSERVATION_FORMAT: Final[str] = \
    '(A5, A7, A1, A1, A1, I4, 1X, I2, 1X, F9.6, I2, 1X, I2, 1X, F6.3, I3, 1X, I2, 1X, F5.2, 9X, F5.2, A1, 6X, A3)'
observation_reader = FortranRecordReader(OBSERVATION_FORMAT)


def parse_observation_line(body: MinorPlanet, line: str) -> MinorPlanetObservation:
    packed_num, packed_desig, discov_ast, note1, note2, year, month, day_dec, ra_hour, ra_minute, ra_second, \
        decl_degree, decl_minute, decl_second, mag, band, observatory_code = observation_reader.read(line)
    observatory = observatories[observatory_code]
    epoch = decimal_day_date_to_time(year, month, day_dec, observatory.location)
    return MinorPlanetObservation.with_band(target_body=body, epoch=epoch, observatory=observatory,
                                            ra=np.radians(15 * ra_hour + ra_minute / 4 + ra_second / 240),
                                            dec=np.radians(decl_degree + decl_minute / 60 + decl_second / 3600),
                                            observed_magnitude=mag, band=band)


def parse_observations(body: MinorPlanet,
                       accept_methods: Collection[str] | None = ('B', 'C')) -> list[MinorPlanetObservation]:
    if not observatories:
        load_observatories()
    result: list[MinorPlanetObservation] = []
    with open(PROJECT_ROOT + body.observations_filepath, 'rt', encoding='utf-8') as file:
        prev_time = None
        for line in file.readlines():
            if accept_methods is not None:
                if line[14] not in accept_methods:
                    continue
            else:
                if not line[14].isupper():
                    continue
            obs = parse_observation_line(body, line)
            if obs.epoch == prev_time:
                continue
            if not line[65:71].strip():
                continue
            result.append(obs)
            prev_time = obs.epoch
    return result


'''
def parse_ephemeris(body: MinorPlanet) -> dict[float, MinorPlanetEphemeris]:
    data: dict[float, MinorPlanetEphemeris] = {}
    with open(PROJECT_ROOT + body.ephemeris_filepath, 'rt', encoding='utf-8') as file:
        for line in file.readlines():
            (t, bx, by, bz, lon, lat, h, ox, oy, oz, sx, sy, sz, a, e, i, n, asc_long, peri_arg, mm, v,
             ra, dec, app_ra, app_dec, vv, th, phi) = line.split(' ')
            time = Time(t, format='decimalyear', scale='utc')
            body_pos = CartesianRepresentation(float(bx), float(by), float(bz), u.au)
            obs_loc = EarthLocation.from_geodetic(float(lon) * u.rad, float(lat) * u.rad, float(h) * u.m)
            obs_pos = CartesianRepresentation(float(ox), float(oy), float(oz), u.au)
            sun_pos = CartesianRepresentation(float(sx), float(sy), float(sz), u.au)
            data[float(t)] = MinorPlanetEphemeris(body, time, body_pos, obs_loc, obs_pos, sun_pos, float(a) * u.au,
                                                  float(e) * u.dimensionless_unscaled, float(i) * u.rad,
                                                  float(n) * u.rad / u.year, float(asc_long) * u.rad,
                                                  float(peri_arg) * u.rad, float(mm) * u.rad, float(v) * u.rad,
                                                  float(ra) * u.rad, float(dec) * u.rad, float(app_ra) * u.rad,
                                                  float(app_dec) * u.rad, float(vv) * u.mag, float(th) * u.rad,
                                                  float(phi) * u.rad)
    return data


def dump_ephemeris(ephs: dict[float, MinorPlanetEphemeris], body: MinorPlanet) -> None:
    with open(PROJECT_ROOT + body.ephemeris_filepath, 'wt', encoding='utf-8') as file:
        lines = []
        for t, eph in sorted(ephs.items()):
            bx, by, bz = eph.body_position.get_xyz().to_value(u.au)
            lon, lat, h = eph.observer_location.geodetic
            lon = lon.to_value(u.rad)
            lat = lat.to_value(u.rad)
            h = h.to(u.m).value
            ox, oy, oz = eph.observer_position.get_xyz().to_value(u.au)
            sx, sy, sz = eph.sun_position.get_xyz().to_value(u.au)
            a = eph.semi_major_axis.to_value(u.au)
            e = eph.eccentricity.value
            i = eph.inclination.to_value(u.rad)
            n = eph.mean_motion.to_value(u.rad / u.year)
            asc_long = eph.ascending_longitude.to_value(u.rad)
            peri_arg = eph.periapsis_argument.to_value(u.rad)
            mm = eph.mean_anomaly.to_value(u.rad)
            v = eph.true_anomaly.to_value(u.rad)
            ra = eph.right_ascension.to_value(u.rad)
            dec = eph.declination.to_value(u.rad)
            app_ra = eph.apparent_right_ascension.to_value(u.rad)
            app_dec = eph.apparent_declination.to_value(u.rad)
            vv = eph.visual_magnitude.to_value(u.mag)
            th = eph.elongation.to_value(u.rad)
            phi = eph.phase.to_value(u.rad)
            lines.append(f'{t} {bx} {by} {bz} {lon} {lat} {h} {ox} {oy} {oz} {sx} {sy} {sz} {a} {e} {i} {n} '
                         f'{asc_long} {peri_arg} {mm} {v} {ra:.6f} {dec:.6f} {app_ra:.6f} {app_dec:.6f} '
                         f'{vv} {th} {phi}\n')

        file.writelines(lines)
'''
