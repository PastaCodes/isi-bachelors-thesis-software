import calendar
import datetime as dt
import math
from typing import Final

import numpy as np
import scipy
from astropy.units import Quantity
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from fortranformat import FortranRecordReader


# Rough estimate of the standard deviation based on the number of provided decimals
def sd(decimals: str) -> float:
    precision = len(decimals.replace(' ', ''))
    return 0.5 * 10 ** -precision


class DecimalDayDatetime:
    def __init__(self, year: int, month: int, day_dec: float) -> None:
        self.year = year
        self.month = month
        self.day_dec = day_dec

    def to_datetime(self) -> dt.datetime:
        day_frac, day = math.modf(self.day_dec)
        return dt.datetime(self.year, self.month, int(day)) + dt.timedelta(days=day_frac)

    def to_decimal_year(self) -> float:
        day_frac, day = math.modf(self.day_dec)
        year_day_dec = dt.datetime(self.year, self.month, int(day)).timetuple().tm_yday + day_frac
        return self.year + year_day_dec / (366 if calendar.isleap(self.year) else 365)


# As described by https://www.minorplanetcenter.net/iau/info/OpticalObs.html
class MinorPlanetObservation:
    FORMAT: Final[str] = \
        '(A5, A7, A1, A1, A1, I4, 1X, I2, 1X, F9.6, I2, 1X, I2, 1X, F6.3, I3, 1X, I2, 1X, F5.2, 9X, F5.2, A1, 6X, A3)'
    reader = FortranRecordReader(FORMAT)

    def __init__(self, datetime: DecimalDayDatetime, year_sd: float, right_ascension: float, right_ascension_sd: float,
                 declination: float, declination_sd: float, magnitude: float, magnitude_sd: float, band: str,
                 is_discovery: bool) -> None:
        self.datetime = datetime
        self.year_sd = year_sd
        self.right_ascension = right_ascension
        self.right_ascension_sd = right_ascension_sd
        self.declination = declination
        self.declination_sd = declination_sd
        self.magnitude = magnitude
        self.magnitude_sd = magnitude_sd
        self.band = band
        self.is_discovery = is_discovery

    @classmethod
    def from_line(cls, line: str) -> 'MinorPlanetObservation':
        packed_num, packed_desig, discov_ast, note1, note2, year, month, day_dec, ra_hour, ra_minute, ra_second, \
            decl_degree, decl_minute, decl_second, mag, band, observatory = cls.reader.read(line)
        return cls(datetime=DecimalDayDatetime(year, month, day_dec),
                   year_sd=(sd(line[27:32]) / 365.25),
                   right_ascension=(ra_hour + ra_minute / 60 + ra_second / 3600),
                   right_ascension_sd=sd(line[42:44]),
                   declination=(decl_degree + decl_minute / 60 + decl_second / 3600),
                   declination_sd=sd(line[55:56]),
                   magnitude=mag,
                   magnitude_sd=sd(line[69:70]),
                   band=band,
                   is_discovery=(discov_ast == '*'))


def parse_file(path: str, accept_methods: list[str] | None = None) -> list[MinorPlanetObservation]:
    with open(path, 'rt', encoding='utf-8') as file:
        return [MinorPlanetObservation.from_line(line)
                for line in file.readlines()
                if (accept_methods is not None and line[14] in accept_methods) or line[14].isupper()]


def elongation_from_observation(right_ascension: float, declination: float, earth_sun_position: np.ndarray,
                                earth_sun_distance: float | None = None) -> float:
    if earth_sun_distance is None:
        earth_sun_distance = np.linalg.norm(earth_sun_position)
    cos_a = math.cos(right_ascension)
    sin_a = math.sin(right_ascension)
    cos_d = math.cos(declination)
    sin_d = math.sin(declination)
    x, y, z = earth_sun_position
    num = x * cos_d * cos_a + y * cos_d * sin_a + z * sin_d
    return math.acos(-num / earth_sun_distance)


def phi(phase: float, slope: float) -> float:
    half_tan = math.tan(phase / 2)
    return (1 - slope) * math.exp(-3.33 * half_tan ** 0.63) + slope * math.exp(-1.87 * half_tan ** 1.22)


def phi_prime(phase: float, slope: float) -> float:
    half_tan = math.tan(phase / 2)
    half_sec = 1 / math.cos(phase / 2)
    return -(half_sec ** 2) * (1.05 * (1 - slope) * half_tan ** -0.37 * math.exp(-3.33 * half_tan ** 0.63) +
                               1.14 * slope * half_tan ** 0.22 * math.exp(-1.87 * half_tan ** 1.22))


# https://www.minorplanetcenter.net/iau/info/BandConversion.txt
VISUAL_CORRECTION: Final[dict[str, float]] = \
    {' ': -0.8, 'U': -1.3, 'B': -0.8, 'g': -0.35, 'V': 0, 'r': 0.14, 'R': 0.4, 'C': 0.4, 'W': 0.4, 'i': 0.32,
     'z': 0.26, 'I': 0.8, 'J': 1.2, 'w': -0.13, 'y': 0.32, 'L': 0.2, 'H': 1.4, 'K': 1.7, 'Y': 0.7, 'G': 0.28, 'v': 0,
     'c': -0.05, 'o': 0.33, 'u': 2.5}


def distance_from_magnitude(observed_magnitude: float, band: str, elongation: float, earth_sun_distance: float,
                            phase_guess: float, absolute_magnitude: float, slope: float) -> float:
    visual_magnitude = observed_magnitude + VISUAL_CORRECTION[band]
    mag_diff = visual_magnitude - absolute_magnitude
    elong_sin = math.sin(elongation)

    def f(phase: float) -> float:
        return ((earth_sun_distance ** 2) * elong_sin * math.sin(phase + elongation) -
                (10 ** (0.2 * mag_diff)) * (math.sin(phase) ** 2) * math.sqrt(phi(phase, slope)))

    def f_prime(phase: float) -> float:
        return ((earth_sun_distance ** 2) * elong_sin * math.cos(phase + elongation) - (10 ** (0.2 * mag_diff)) *
                math.sin(phase) * math.cos(phase) * (phi(phase, slope)) ** -0.5 * phi_prime(phase, slope))

    phase_root = scipy.optimize.newton(f, phase_guess, f_prime)
    return earth_sun_distance * math.sin(phase_root + elongation) / math.sin(phase_root)


def output_transform(obs: MinorPlanetObservation, absolute_magnitude: float, slope: float = 0.15) -> np.ndarray:
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
    solar_system_ephemeris.set('jpl')
    obstime = Time(obs.datetime.to_datetime(), scale='utc')
    earth_position: CartesianRepresentation = get_body_barycentric('earth', obstime)
    sun_position: CartesianRepresentation = get_body_barycentric('sun', obstime)
    earth_sun_position: CartesianRepresentation = earth_position - sun_position
    earth_sun_distance: Quantity = earth_sun_position.norm()
    elongation = elongation_from_observation(obs.right_ascension, obs.declination,
                                             earth_sun_position.get_xyz().to(u.au).value,
                                             earth_sun_distance.to(u.au).value)
    phase_guess = 1  # TODO
    distance = distance_from_magnitude(obs.magnitude, obs.band, elongation, earth_sun_distance.to(u.au).value,
                                       phase_guess, absolute_magnitude, slope)
    observed = SkyCoord(ra=(obs.right_ascension * u.hourangle), dec=(obs.declination * u.degree),
                        distance=(distance * u.au), obstime=obstime, frame='fk5')
    state: CartesianRepresentation = observed.icrs.cartesian + earth_position
    return state.get_xyz().to(u.au).value


def main():
    obs = parse_file('data/Bennu.txt', accept_methods=['B', 'C'])
    data = []
    for o in obs:
        try:
            pos = output_transform(o, 20.9, 0.04)
            # data.append((o.datetime.to_decimal_year(), pos))
            data.append(pos)
        except:
            continue
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    d = np.array(data)
    ax.scatter(d.T[0], d.T[1], d.T[2], c='r', depthshade=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


if __name__ == '__main__':
    main()
