import itertools
from typing import Iterable

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import EarthLocation, CartesianRepresentation
from astropy.time import Time
from astropy.units import Quantity

from main import MinorPlanetEphemeris, MinorPlanet
from parse import parse_observations


def query_vector_table(command: str, times: list[Time]) -> Iterable[CartesianRepresentation]:
    tlist = '%20'.join([f'%27{t.utc.to_value('jd')}%27' for t in times])
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          f'&COMMAND={command}' \
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
        yield CartesianRepresentation(float(columns[2]), float(columns[3]), float(columns[4]), u.au)


def query_orbital_elements(command: str, times: list[Time]) -> \
        Iterable[tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]]:
    tlist = '%20'.join([f'%27{t.tdb.to_value('jd')}%27' for t in times])
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          f'&COMMAND={command}' \
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
        e = float(columns[2]) * u.dimensionless_unscaled
        i = float(columns[4]) * u.deg
        asc_long = float(columns[5]) * u.deg
        peri_arg = float(columns[6]) * u.deg
        n = float(columns[8]) * u.deg / u.d
        mm = float(columns[9]) * u.deg
        v = float(columns[10]) * u.deg
        a = float(columns[11]) * u.au
        yield a, e, i, n, asc_long, peri_arg, mm, v


def query_observer_table(command: str, times: list[Time], location: EarthLocation) -> \
        Iterable[tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]]:
    tlist = '%20'.join([f'%27{t.utc.to_value('jd')}%27' for t in times])
    lon, lat, h = location.geodetic
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          f'&COMMAND={command}' \
          '&MAKE_EPHEM=YES' \
          '&EPHEM_TYPE=OBSERVER' \
          '&CENTER=coord%40399' \
          '&COORD_TYPE=GEODETIC' \
          f'&SITE_COORD=%27{lon.to(u.deg).value}%2C{lat.to(u.deg).value}%2C{h.to(u.km).value}%27' \
          '&TLIST_TYPE=JD' \
          '&TIME_TYPE=UT' \
          '&QUANTITIES=%271%2C2%2C9%2C23%2C24%2C28%2C43%27' \
          '&REF_SYSTEM=ICRF' \
          '&REF_PLANE=FRAME' \
          '&ANG_FORMAT=DEG' \
          '&RANGE_UNITS=AU' \
          '&APPARENT=REFRACTED' \
          '&EXTRA_PREC=YES' \
          '&CSV_FORMAT=YES' \
          '&TLIST=' + tlist

    response = requests.get(url)
    content = response.text.split('$$SOE\n', 1)[1].split('\n$$EOE', 1)[0]
    for line in content.split('\n'):
        columns = line.split(',')
        ra = float(columns[3]) * u.deg
        dec = float(columns[4]) * u.deg
        app_ra = float(columns[5]) * u.deg
        app_dec = float(columns[6]) * u.deg
        vv = float(columns[7]) * u.mag
        th = np.radians(float(columns[9])) * u.deg
        phi = np.radians(float(columns[13])) * u.deg
        yield ra, dec, app_ra, app_dec, vv, th, phi


def query_ephemeris(body: MinorPlanet, max_batch_size: int = 30) -> dict[float, MinorPlanetEphemeris]:
    command = f'%27DES%3D{body.name.replace(' ', '%20')}%3B%27'
    all_observations = parse_observations(body)
    by_observatory = itertools.groupby(all_observations, lambda obs: obs.observatory)
    result: dict[float, MinorPlanetEphemeris] = {}
    for observatory, group in by_observatory:
        batches = itertools.batched(group, max_batch_size)
        for batch in batches:
            times = [o.epoch for o in batch]
            for epoch, position, orb_data, obs_data in zip(times,
                                                           query_vector_table(command, times),
                                                           query_orbital_elements(command, times),
                                                           query_observer_table(command, times, observatory.location)):
                a, e, i, n, asc_long, peri_arg, mm, v = orb_data
                ra, dec, app_ra, app_dec, vv, th, phi = obs_data
                t = epoch.to_value('decimalyear')
                result[t] = MinorPlanetEphemeris(body, epoch, observatory.location, position, a, e, i, n,
                                                 asc_long, peri_arg, mm, v, ra, dec, app_ra, app_dec, vv, th, phi)

            print(f'Batch completed. Current total: {len(result)} of {len(all_observations)}')

    return result
