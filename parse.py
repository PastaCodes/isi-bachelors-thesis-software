from typing import Final

import numpy as np
from fortranformat import FortranRecordReader

from main import MinorPlanetObservation, MinorPlanet, MinorPlanetEphemeris, OBJ_BENNU
from misc import DecimalDayDatetime, sd


# As described in https://www.minorplanetcenter.net/iau/info/OpticalObs.html
FORMAT: Final[str] = \
    '(A5, A7, A1, A1, A1, I4, 1X, I2, 1X, F9.6, I2, 1X, I2, 1X, F6.3, I3, 1X, I2, 1X, F5.2, 9X, F5.2, A1, 6X, A3)'
reader = FortranRecordReader(FORMAT)


def parse_obs_line(obj: MinorPlanet, line: str) -> MinorPlanetObservation:
    packed_num, packed_desig, discov_ast, note1, note2, year, month, day_dec, ra_hour, ra_minute, ra_second, \
        decl_degree, decl_minute, decl_second, mag, band, observatory = reader.read(line)
    return MinorPlanetObservation(obj=obj,
                                  datetime=DecimalDayDatetime(year, month, day_dec),
                                  year_sd=(sd(line[27:32]) / 365.25),
                                  right_ascension=(ra_hour + ra_minute / 60 + ra_second / 3600),
                                  right_ascension_sd=sd(line[42:44]),
                                  declination=(decl_degree + decl_minute / 60 + decl_second / 3600),
                                  declination_sd=sd(line[55:56]),
                                  magnitude=mag,
                                  magnitude_sd=sd(line[69:70]),
                                  band=band,
                                  is_discovery=(discov_ast == '*'))


def parse_obs_file(obj: MinorPlanet, accept_methods: list[str] | None = None) -> list[MinorPlanetObservation]:
    result: list[MinorPlanetObservation] = []
    with open(obj.observations_file, 'rt', encoding='utf-8') as file:
        prev_time = None
        for line in file.readlines():
            if accept_methods is not None:
                if line[14] not in accept_methods:
                    continue
            else:
                if not line[14].isupper():
                    continue
            obs = parse_obs_line(obj, line)
            if obs.obstime == prev_time:
                continue
            if not line[65:71].strip():
                continue
            result.append(obs)
            prev_time = obs.obstime
    return result


def dump_ephemeris(ephs: dict[float, MinorPlanetEphemeris], file_path: str) -> None:
    with open(file_path, 'wt', encoding='utf-8') as file:
        lines = [f'{t} {eph.position[0]} {eph.position[1]} {eph.position[2]} {eph.semi_major_axis} {eph.eccentricity} '
                 f'{eph.inclination} {eph.mean_motion} {eph.mean_anomaly} {eph.true_anomaly} {eph.right_ascension:.6f} '
                 f'{eph.declination:.6f} {eph.apparent_ra:.6f} {eph.apparent_decl:.6f} {eph.visual_magnitude} '
                 f'{eph.elongation} {eph.phase}\n' for t, eph in ephs.items()]
        file.writelines(lines)


def load_ephemeris(obj: MinorPlanet, file_path: str) -> dict[float, MinorPlanetEphemeris]:
    data: dict[float, MinorPlanetEphemeris] = {}
    with open(file_path, 'rt', encoding='utf-8') as file:
        for line in file.readlines():
            t, x, y, z, a, e, i, n, m, v, ra, decl, app_ra, app_decl, vv, th, phi = line.split(' ')
            data[float(t)] = MinorPlanetEphemeris(obj, np.array([float(x), float(y), float(z)]), float(a), float(e),
                                                  float(i), float(n), float(m), float(v), float(ra), float(decl),
                                                  float(app_ra), float(app_decl), float(vv), float(th), float(phi))
    return data
