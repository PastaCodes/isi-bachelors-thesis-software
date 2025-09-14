import itertools
import urllib.parse
from typing import Iterable

import astropy.units as u
import requests
from astropy.coordinates import EarthLocation
from astropy.time import Time

from classes import MinorPlanet
from misc import PROJECT_ROOT
from parse import parse_observations


def query_target_position(command: str, times: list[Time]) -> Iterable[tuple[str, str, str]]:
    print('Querying target position...')
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
        x = columns[2].strip()
        y = columns[3].strip()
        z = columns[4].strip()
        yield x, y, z


def query_orbital_elements(command: str, times: list[Time]) -> Iterable[tuple[str, str, str, str, str, str, str, str]]:
    print('Querying orbital elements...')
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
        e = columns[2].strip()
        i = columns[4].strip()
        om = columns[5].strip()
        w = columns[6].strip()
        n = columns[8].strip()
        mm = columns[9].strip()
        v = columns[10].strip()
        a = columns[11].strip()
        yield a, e, i, om, w, mm, v, n


def query_observation(command: str, times: list[Time], location: EarthLocation) -> \
        Iterable[tuple[str, str, str, str, str, str, str, str]]:
    print('Querying observation data...')
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
          '&QUANTITIES=%271%2C2%2C9%2C23%2C24%2C43%27' \
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
        astro_ra = columns[3].strip()
        astro_dec = columns[4].strip()
        app_ra = columns[5].strip()
        app_dec = columns[6].strip()
        app_vv = columns[7].strip()
        app_th = columns[9].strip()
        app_phi = columns[11].strip()
        astro_phi = columns[12].strip()
        yield astro_ra, astro_dec, app_ra, app_dec, app_vv, app_th, app_phi, astro_phi


def query_observer_position(location: EarthLocation, times: list[Time]) -> Iterable[tuple[str, str, str]]:
    print('Querying observer position...')
    tlist = '%20'.join([f'%27{t.utc.to_value('jd')}%27' for t in times])
    lon, lat, h = location.geodetic
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          f'&COMMAND=%27g%3A{lon.to(u.deg).value}%2C{lat.to(u.deg).value}%2C{h.to(u.km).value}%40399%27' \
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
        x = columns[2].strip()
        y = columns[3].strip()
        z = columns[4].strip()
        yield x, y, z


def query_sun_position(times: list[Time]) -> Iterable[tuple[str, str, str]]:
    print('Querying Sun position...')
    tlist = '%20'.join([f'%27{t.utc.to_value('jd')}%27' for t in times])
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api' \
          '?format=text' \
          f'&COMMAND=10' \
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
        x = columns[2].strip()
        y = columns[3].strip()
        z = columns[4].strip()
        yield x, y, z


def query_ephemeris(target_body: MinorPlanet, max_batch_size: int = 30):
    command = f'%27DES%3D{urllib.parse.quote(target_body.jpl_designation)}%3B%27'
    all_observations = list(parse_observations(target_body))
    by_observatory = itertools.groupby(all_observations, lambda obs: obs.observatory)
    ephs: dict[float, tuple] = dict()
    for observatory, group in by_observatory:
        batches = itertools.batched(group, max_batch_size)
        for batch in batches:
            times = [o.epoch for o in batch]
            for epoch, tgt_pos, orb_data, obs_data, obs_pos, sun_pos in \
                    zip(times,
                        query_target_position(command, times),
                        query_orbital_elements(command, times),
                        query_observation(command, times, observatory.location),
                        query_observer_position(observatory.location, times),
                        query_sun_position(times)):
                t = epoch.tdb.to_value('jd')
                tx, ty, tz = tgt_pos
                a, e, i, om, w, mm, v, n = orb_data
                astro_ra, astro_dec, app_ra, app_dec, app_vv, app_th, app_phi, astro_phi = obs_data
                ox, oy, oz = obs_pos
                sx, sy, sz = sun_pos
                ephs[t] = (tx, ty, tz, ox, oy, oz, sx, sy, sz, a, e, i, om, w, mm, v, n, astro_ra, astro_dec,
                           app_ra, app_dec, app_vv, app_th, app_phi, astro_phi)

            print(f'Batch completed. Current total: {len(ephs)} of {len(all_observations)}')

    lines = []
    for t, eph in sorted(ephs.items()):
        (tx, ty, tz, ox, oy, oz, sx, sy, sz, a, e, i, om, w, mm, v, n, astro_ra, astro_dec,
         app_ra, app_dec, app_vv, app_th, app_phi, astro_phi) = eph
        lines.append(f'{t:.10f} {tx} {ty} {tz} {ox} {oy} {oz} {sx} {sy} {sz} {a} {e} {i} {om} {w} {mm} {v} {n} '
                     f'{astro_ra} {astro_dec} {app_ra} {app_dec} {app_vv} {app_th} {app_phi} {astro_phi}\n')

    with open(PROJECT_ROOT + target_body.ephemeris_filepath, 'wt', encoding='utf-8') as file:
        file.writelines(lines)
