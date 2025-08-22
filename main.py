import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

from classes import MinorPlanet, StateHint

BODY_BENNU = MinorPlanet(name='1999 RQ36',
                         hh=(20.21 * u.mag),
                         gg=(-0.031 * u.dimensionless_unscaled),
                         a=(1.128 * u.au),
                         e=(0.204 * u.dimensionless_unscaled),
                         i=(0.514 * u.rad),
                         observations_filepath='data/Bennu.txt',
                         ephemeris_filepath='data/Bennu_eph.txt',
                         initial_state_hint=StateHint(mean_anomaly=(2 * np.pi * u.rad),
                                                      ascending_longitude=(0 * u.rad)))

BODY_2010_RF12 = MinorPlanet(name='2010 RF12',
                             hh=(28.42 * u.mag),
                             gg=(0.15 * u.dimensionless_unscaled),
                             a=(1.061 * u.au),
                             e=(0.188 * u.dimensionless_unscaled),
                             i=(0.424 * u.rad),
                             observations_filepath='data/2010_rf12.txt',
                             ephemeris_filepath='data/2010_rf12_eph.txt')

BODY_2024_YR4 = MinorPlanet(name='2024 YR4',
                            hh=(23.92 * u.mag),
                            gg=(0.15 * u.dimensionless_unscaled),
                            a=(2.516 * u.au),
                            e=(0.662 * u.dimensionless_unscaled),
                            i=(0.469 * u.rad),
                            observations_filepath='data/2024_yr4.txt',
                            ephemeris_filepath='data/2024_yr4_eph.txt')


if __name__ == '__main__':
    from astropy.time import Time
    print(Time('2451500.266967312', format='jd', scale='tdb').utc.to_value('iso'))
    print(Time('2451500.266967312', format='jd', scale='tdb', location=EarthLocation.from_geodetic(12.49222, 41.89028)).utc.to_value('iso'))
