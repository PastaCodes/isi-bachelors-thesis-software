from typing import Final, Collection

import astropy.units as u
from astropy.coordinates import EarthLocation, CartesianRepresentation
from astropy.time import Time
from fortranformat import FortranRecordReader

from main import MinorPlanetObservation, MinorPlanet, MinorPlanetEphemeris, Observatory
from misc import location_from_parallax, decimal_day_date_to_time

# As described in https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html
OBSERVATORY_FORMAT: Final[str] = '(A3, 1X, F9.5, F8.6, F9.6, A50)'
observatory_reader = FortranRecordReader(OBSERVATORY_FORMAT)
observatories_file_path = 'data/ObsCodes.txt'
observatories: dict[str, Observatory] = {}


def load_observatories():
    with open(observatories_file_path, 'rt', encoding='utf-8') as file:
        for line in file.readlines()[1:]:
            if not line[6]:  # No location data; likely not on Earth
                continue
            code, lon, p1, p2, name = observatory_reader.read(line)
            loc = location_from_parallax(float(lon) * u.deg, float(p1) * u.earthRad, float(p2) * u.earthRad)
            observatories[code] = Observatory(code, name, loc)


# As described in https://www.minorplanetcenter.net/iau/info/OpticalObs.html
OBSERVATION_FORMAT: Final[str] = \
    '(A5, A7, A1, A1, A1, I4, 1X, I2, 1X, F9.6, I2, 1X, I2, 1X, F6.3, I3, 1X, I2, 1X, F5.2, 9X, F5.2, A1, 6X, A3)'
observation_reader = FortranRecordReader(OBSERVATION_FORMAT)


def parse_observation_line(body: MinorPlanet, line: str) -> MinorPlanetObservation:
    packed_num, packed_desig, discov_ast, note1, note2, year, month, day_dec, ra_hour, ra_minute, ra_second, \
        decl_degree, decl_minute, decl_second, mag, band, observatory_code = observation_reader.read(line)
    return MinorPlanetObservation(body=body,
                                  epoch=decimal_day_date_to_time(year, month, day_dec),
                                  # epoch_var=(sd(line[27:32]) / 365.25),
                                  observatory=observatories[observatory_code],
                                  ra=(ra_hour + ra_minute / 60 + ra_second / 3600) * u.hourangle,
                                  # ra_var=sd(line[42:44]),
                                  dec=(decl_degree + decl_minute / 60 + decl_second / 3600) * u.deg,
                                  # dec_var=sd(line[55:56]),
                                  mag=mag * u.mag,
                                  # mag_var=sd(line[69:70]),
                                  band=band)


def parse_observations(body: MinorPlanet,
                       accept_methods: Collection[str] | None = ('B', 'C')) -> list[MinorPlanetObservation]:
    if not observatories:
        load_observatories()
    result: list[MinorPlanetObservation] = []
    with open(body.observations_filepath, 'rt', encoding='utf-8') as file:
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


def parse_ephemeris(obj: MinorPlanet) -> dict[float, MinorPlanetEphemeris]:
    data: dict[float, MinorPlanetEphemeris] = {}
    with open(obj.ephemeris_filepath, 'rt', encoding='utf-8') as file:
        for line in file.readlines():
            t, x, y, z, lon, lat, h, a, e, i, n, asc_long, peri_arg, mm, v, ra, dec, app_ra, app_dec, vv, th, phi = \
                line.split(' ')
            time = Time(t, format='decimalyear', scale='utc')
            obs_location = EarthLocation.from_geodetic(float(lon) * u.rad, float(lat) * u.rad, float(z) * u.m)
            position = CartesianRepresentation(float(x), float(y), float(z), u.au)
            data[float(t)] = MinorPlanetEphemeris(obj, time, obs_location, position, float(a) * u.au,
                                                  float(e) * u.dimensionless_unscaled, float(i) * u.rad,
                                                  float(n) * u.rad / u.year, float(asc_long) * u.rad,
                                                  float(peri_arg) * u.rad, float(mm) * u.rad, float(v) * u.rad,
                                                  float(ra) * u.rad, float(dec) * u.rad, float(app_ra) * u.rad,
                                                  float(app_dec) * u.rad, float(vv) * u.mag, float(th) * u.rad,
                                                  float(phi) * u.rad)
    return data


def dump_ephemeris(ephs: dict[float, MinorPlanetEphemeris], file_path: str) -> None:
    with open(file_path, 'wt', encoding='utf-8') as file:
        lines = []
        for t, eph in sorted(ephs.items()):
            x, y, z = eph.position.get_xyz().to(u.au).value
            lon, lat, h = eph.observer_location.geodetic
            lon = lon.to(u.rad).value
            lat = lat.to(u.rad).value
            h = h.to(u.m).value
            a = eph.semi_major_axis.to(u.au).value
            e = eph.eccentricity.value
            i = eph.inclination.to(u.rad).value
            n = eph.mean_motion.to(u.rad / u.year).value
            asc_long = eph.ascending_longitude.to(u.rad).value
            peri_arg = eph.periapsis_argument.to(u.rad).value
            mm = eph.mean_anomaly.to(u.rad).value
            v = eph.true_anomaly.to(u.rad).value
            ra = eph.right_ascension.to(u.rad).value
            dec = eph.declination.to(u.rad).value
            app_ra = eph.apparent_right_ascension.to(u.rad).value
            app_dec = eph.apparent_declination.to(u.rad).value
            vv = eph.visual_magnitude.to(u.mag).value
            th = eph.elongation.to(u.rad).value
            phi = eph.phase.to(u.rad).value
            lines.append(f'{t} {x} {y} {z} {lon} {lat} {h} {a} {e} {i} {n} {asc_long} {peri_arg} {mm} {v} '
                         f'{ra:.6f} {dec:.6f} {app_ra:.6f} {app_dec:.6f} {vv} {th} {phi}\n')

        file.writelines(lines)
