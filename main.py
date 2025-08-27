from classes import MinorPlanet


BODY_BENNU = MinorPlanet(display_name='Bennu',
                         jpl_designation='1999 RQ36',
                         observations_filepath='data/Bennu.txt',
                         ephemeris_filepath='data/Bennu_eph.txt')

BODY_2010_RF12 = MinorPlanet(jpl_designation='2010 RF12',
                             observations_filepath='data/2010_rf12.txt',
                             ephemeris_filepath='data/2010_rf12_eph.txt')

BODY_2024_YR4 = MinorPlanet(jpl_designation='2024 YR4',
                            observations_filepath='data/2024_yr4.txt',
                            ephemeris_filepath='data/2024_yr4_eph.txt')


if __name__ == '__main__':
    from filter import do_filter
    do_filter(BODY_BENNU,
              a0=1.128,
              e0=0.204,
              i0=0.514,
              hh0=20.21,
              gg0=-0.031,
              mm0_hint=6.28,
              om0_hint=0.0)
