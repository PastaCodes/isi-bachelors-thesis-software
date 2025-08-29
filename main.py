from classes import MinorPlanet


BODY_BENNU = MinorPlanet(display_name='Bennu',
                         jpl_designation='1999 RQ36',
                         absolute_magnitude=20.21,
                         slope_parameter=-0.031,
                         observations_filepath='data/Bennu.txt',
                         ephemeris_filepath='data/Bennu_eph.txt')

BODY_2010_RF12 = MinorPlanet(jpl_designation='2010 RF12',
                             absolute_magnitude=28.42,
                             observations_filepath='data/2010_rf12.txt',
                             ephemeris_filepath='data/2010_rf12_eph.txt')

BODY_2024_YR4 = MinorPlanet(jpl_designation='2024 YR4',
                            absolute_magnitude=23.92,
                            observations_filepath='data/2024_yr4.txt',
                            ephemeris_filepath='data/2024_yr4_eph.txt')


if __name__ == '__main__':
    from visualize import cherry_picked
    cherry_picked()
