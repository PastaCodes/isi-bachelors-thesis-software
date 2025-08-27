from classes import MinorPlanet


BODY_BENNU = MinorPlanet(('Bennu', '1999 RQ36'), 'data/Bennu.txt', 'data/Bennu_eph.txt')
BODY_2010_RF12 = MinorPlanet('2010 RF12', 'data/2010_rf12.txt', 'data/2010_rf12_eph.txt')
BODY_2024_YR4 = MinorPlanet('2024 YR4', 'data/2024_yr4.txt', 'data/2024_yr4_eph.txt')


if __name__ == '__main__':
    from filter import do_filter
    do_filter(BODY_BENNU,
              dir_conc=1E7,
              vv_var=0.01,
              a0=1.128,
              e0=0.204,
              i0=0.514,
              hh0=20.21,
              gg0=-0.031,
              mm0_hint=6.28,
              om0_hint=0.0)
