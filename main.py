from classes import MinorPlanet


BODY_BENNU = MinorPlanet(display_name='Bennu',
                         jpl_designation='1999 RQ36',
                         observations_filepath='data/Bennu.txt',
                         ephemeris_filepath='data/Bennu_eph.txt',
                         a0=1.128,
                         e0=0.204,
                         i0=0.514,
                         hh0=20.21,
                         gg0=-0.031,
                         mm0_hint=6.28,
                         om0_hint=0.0,
                         mm_var=3.5E-7,
                         a_var=2.9E-8,
                         e_var=1.0E-8,
                         i_var=1.6E-10,
                         om_var=5.0E-11,
                         w_var=1.7E-7,
                         n_var=1.1E-11,
                         dir_var=1.1E-4,
                         vv_var=2.7E-1,
                         dt_exp=1.22)

BODY_2010_RF12 = MinorPlanet(jpl_designation='2010 RF12',
                             observations_filepath='data/2010_rf12.txt',
                             ephemeris_filepath='data/2010_rf12_eph.txt',
                             a0=1.061,
                             e0=0.188,
                             i0=0.424,
                             hh0=28.42,
                             gg0=0.15,
                             mm0_hint=6.28,
                             om0_hint=0.0,
                             mm_var=5.8E-6,
                             a_var=1.5E-7,
                             e_var=2.4E-7,
                             i_var=4.1E-8,
                             om_var=1.6E-8,
                             w_var=2.0E-5,
                             n_var=9.7E-11,
                             dir_var=4.0E-5,
                             vv_var=1.2E-1,
                             dt_exp=0.93)

BODY_2024_YR4 = MinorPlanet(jpl_designation='2024 YR4',
                            observations_filepath='data/2024_yr4.txt',
                            ephemeris_filepath='data/2024_yr4_eph.txt',
                            a0=2.516,
                            e0=0.662,
                            i0=0.469,
                            hh0=23.92,
                            gg0=0.15,
                            mm0_hint=0.0,
                            om0_hint=6.28,
                            mm_var=6.7E-4,
                            a_var=4.1E-2,
                            e_var=7.3E-4,
                            i_var=4.1E-6,
                            om_var=4.1E-4,
                            w_var=1.7E-3,
                            n_var=3.1E-7,
                            dir_var=1.1E-5,
                            vv_var=1.0E-1,
                            dt_exp=3.36)


def main() -> None:
    from visualize import plot_errors_bennu, plot_3d_bennu
    plot_3d_bennu()
    plot_errors_bennu()


if __name__ == '__main__':
    main()
