import datetime as dt
import math
from typing import Final

from fortranformat import FortranRecordReader


def sd(decimals: str) -> int:
    """
    Provides a rough estimate of the standard deviation of a measurement based on its provided decimals
    """
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


# As described by https://www.minorplanetcenter.net/iau/info/OpticalObs.html
class MinorPlanetObservation:
    FORMAT: Final[str] = '(A5, A7, A1, A1, A1, I4, 1X, I2, 1X, F9.6, A12, A12, 9X, F5.2, A1, 6X, A3)'
    reader = FortranRecordReader(FORMAT)

    def __init__(self,
                 datetime: DecimalDayDatetime, year_sd: float,
                 right_ascension: str, right_ascension_sd: float,
                 declination: str, declination_sd: float,
                 magnitude: float, magnitude_sd: float, band: str,
                 is_discovery: bool,
                 ) -> None:
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
        packed_num, packed_desig, discov_ast, note1, note2, year, month, day_dec, ra, decl, mag, band, observatory = \
            cls.reader.read(line)
        return cls(
            datetime=DecimalDayDatetime(year, month, day_dec),
            year_sd=(sd(line[27:32]) / 365.25),
            right_ascension=ra.strip(),
            right_ascension_sd=sd(line[42:44]),
            declination=decl.strip(),
            declination_sd=sd(line[55:56]),
            magnitude=mag,
            magnitude_sd=sd(line[69:70]),
            band=band,
            is_discovery=(discov_ast == '*')
        )


def parse_file(path: str, accept_methods: list[str] | None = None) -> list[MinorPlanetObservation]:
    with open(path, 'rt', encoding='utf-8') as file:
        return [
            MinorPlanetObservation.from_line(line)
            for line in file.readlines()
            if (accept_methods is not None and line[14] in accept_methods) or line[14].isupper()
        ]


# https://www.minorplanetcenter.net/iau/info/BandConversion.txt
VISUAL_CORRECTION: Final[dict[str, float]] = {
    ' ': -0.8, 'U': -1.3, 'B': -0.8, 'g': -0.35, 'V': 0, 'r': 0.14, 'R': 0.4, 'C': 0.4, 'W': 0.4, 'i': 0.32, 'z': 0.26,
    'I': 0.8, 'J': 1.2, 'w': -0.13, 'y': 0.32, 'L': 0.2, 'H': 1.4, 'K': 1.7, 'Y': 0.7, 'G': 0.28, 'v': 0, 'c': -0.05,
    'o': 0.33, 'u': 2.5,
}


def phi(phase: float, slope: float) -> float:
    half_tan = math.tan(phase / 2)
    return (1 - slope) * math.exp(-3.33 * half_tan ** 0.63) + slope * math.exp(-1.87 * half_tan ** 1.22)


def distance_from_magnitude(observed_magnitude: float,
                            band: str,
                            elongation: float,
                            earth_sun_distance: float,
                            phase_guess: float,
                            absolute_magnitude: float,
                            slope: float,
                            ) -> float:
    visual_magnitude = observed_magnitude + VISUAL_CORRECTION[band]
    phase = phase_guess
    mag_diff = visual_magnitude - absolute_magnitude
    elong_sin = math.sin(elongation)

    phase_sin = math.sin(phase)
    sum_sin = math.sin(phase + elongation)
    phi_sqrt = math.sqrt(phi(phase, slope))
    phase = (earth_sun_distance ** 2) * elong_sin * sum_sin - (10 ** (0.2 * mag_diff)) * (phase_sin ** 2) * phi_sqrt

    phase_sin = math.sin(phase)
    sum_sin = math.sin(phase + elongation)
    return earth_sun_distance * sum_sin / phase_sin


def output_transform(obs: MinorPlanetObservation, absolute_magnitude: float, slope: float = 0.15):
    from astropy.coordinates import SkyCoord
    from astropy.units import Unit
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
    solar_system_ephemeris.set('jpl')
    obstime = Time(obs.datetime.to_datetime(), scale='utc')
    earth_position = get_body_barycentric('earth', obstime)
    elongation = 0  # todo from obs.right_ascension, obs.declination, and earth_position
    earth_sun_distance = 0  # todo from earth_position
    phase_guess = 0  # todo
    distance = distance_from_magnitude(obs.magnitude, obs.band, elongation, earth_sun_distance, phase_guess,
                                       absolute_magnitude, slope)
    observed = SkyCoord(
        ra=obs.right_ascension,
        dec=obs.declination,
        unit=('hourangle', 'deg'),
        distance=(distance * Unit('au')),
        obstime=obstime,
        frame='fk5',
    )
    state = observed.icrs.cartesian + earth_position
    print(state)
    return state


def plot_observations(obs: list[MinorPlanetObservation]) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.array([o.year_f for o in obs])
    ra = np.array([o.right_ascension for o in obs])
    decl = np.array([o.declination for o in obs])
    plt.figure()
    plt.subplot(121)
    plt.plot(t, ra, 'ro')
    plt.subplot(122)
    plt.plot(t, decl, 'go')
    plt.show()


def main():
    obs = parse_file('data/2024_yr4.txt', accept_methods=['B', 'C'])
    output_transform(obs[0], 23.92, 0.15)


if __name__ == '__main__':
    main()
