import itertools
from typing import Generator

import numpy as np
import pandas as pd

import requests

from main import OBJ_BENNU, MinorPlanetEphemeris


def query_vector_table(utc_times: list[float]) -> Generator[np.ndarray, None, None]:
    tlist = '%20'.join([f'%27{t}%27' for t in utc_times])
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          '&COMMAND=%27DES%3D20101955%3B%27' \
          '&MAKE_EPHEM=YES' \
          '&EPHEM_TYPE=VECTORS' \
          '&CENTER=500%400' \
          '&TLIST_TYPE=JD' \
          '&TIME_TYPE=UT' \
          '&VEC_TABLE=1' \
          '&REF_SYSTEM=ICRF' \
          '&REF_PLANE=FRAME' \
          '&OUT_UNITS=AU-D' \
          '&CSV_FORMAT=YES' \
          '&TLIST=' + tlist

    response = requests.get(url)
    content = response.text.split('$$SOE\n', 1)[1].split('\n$$EOE', 1)[0]
    for line in content.split('\n'):
        columns = line.split(',')
        x, y, z = float(columns[2]), float(columns[3]), float(columns[4])
        yield np.array([x, y, z])


def query_orbital_elements(tbd_times: list[float]) -> \
        Generator[tuple[float, float, float, float, float, float], None, None]:
    tlist = '%20'.join([f'%27{t}%27' for t in tbd_times])
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          '&COMMAND=%27DES%3D20101955%3B%27' \
          '&MAKE_EPHEM=YES' \
          '&EPHEM_TYPE=ELEMENTS' \
          '&CENTER=500%400' \
          '&TLIST_TYPE=JD' \
          '&TIME_TYPE=TDB' \
          '&REF_SYSTEM=ICRF' \
          '&REF_PLANE=FRAME' \
          '&OUT_UNITS=AU-D' \
          '&CSV_FORMAT=YES' \
          '&TLIST=' + tlist

    response = requests.get(url)
    content = response.text.split('$$SOE\n', 1)[1].split('\n$$EOE', 1)[0]
    for line in content.split('\n'):
        columns = line.split(',')
        eccentricity = float(columns[2])
        inclination = float(columns[4]) * np.pi / 180
        mean_motion = float(columns[8]) * 365.25 * np.pi / 180  # From degrees per day to radians per year
        mean_anomaly = float(columns[9]) * np.pi / 180
        true_anomaly = float(columns[10]) * np.pi / 180
        semi_major_axis = float(columns[11])
        yield semi_major_axis, eccentricity, inclination, mean_motion, mean_anomaly, true_anomaly


def query_observer_table(utc_times: list[float]) -> \
        Generator[tuple[float, float, float, float, float, float, float], None, None]:
    tlist = '%20'.join([f'%27{t}%27' for t in utc_times])
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          '&COMMAND=%27DES%3D20101955%3B%27' \
          '&MAKE_EPHEM=YES' \
          '&EPHEM_TYPE=OBSERVER' \
          '&CENTER=500%40399' \
          '&TLIST_TYPE=JD' \
          '&TIME_TYPE=UT' \
          '&QUANTITIES=%271%2C2%2C9%2C23%2C24%2C28%2C43%27' \
          '&REF_SYSTEM=ICRF' \
          '&REF_PLANE=FRAME' \
          '&ANG_FORMAT=DEG' \
          '&RANGE_UNITS=AU' \
          '&APPARENT=REFRACTED' \
          '&CSV_FORMAT=YES' \
          '&TLIST=' + tlist

    response = requests.get(url)
    content = response.text.split('$$SOE\n', 1)[1].split('\n$$EOE', 1)[0]
    for line in content.split('\n'):
        columns = line.split(',')
        right_ascension = float(columns[3]) / 15  # From degrees to hour angles
        declination = float(columns[4])
        apparent_ra = float(columns[5]) / 15
        apparent_decl = float(columns[6])
        visual_magnitude = float(columns[7])
        elongation = float(columns[9]) * np.pi / 180
        phase = float(columns[13]) * np.pi / 180
        yield right_ascension, declination, apparent_ra, apparent_decl, visual_magnitude, elongation, phase


def query_ephemeris(batch_size: int = 30) -> dict[float, MinorPlanetEphemeris]:
    obss = OBJ_BENNU.load_observations()
    times = pd.unique(pd.Series([(o.obstime, o.datetime.to_decimal_year()) for o in obss]))

    result: dict[float, MinorPlanetEphemeris] = {}

    batches = itertools.batched(times, batch_size)
    for batch in batches:
        utc_times = [t[0].to_value('jd') for t in batch]
        tdb_times = [t[0].tdb.to_value('jd') for t in batch]

        for t, position, orb_data, obs_data in zip(batch,
                                                   query_vector_table(utc_times),
                                                   query_orbital_elements(tdb_times),
                                                   query_observer_table(utc_times)):
            a, e, i, n, m, v = orb_data
            ra, decl, app_ra, app_decl, vv, th, phi = obs_data
            result[t[1]] = MinorPlanetEphemeris(OBJ_BENNU, position, a, e, i, n, m, v, ra, decl,
                                                app_ra, app_decl, vv, th, phi)

        print('Batch complete')

    return result
