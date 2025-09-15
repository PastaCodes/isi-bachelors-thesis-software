def main() -> None:
    # Here are some things you can do...

    # --- Query ephemeris data for new body
    # from bodies import BODY_XYZ
    # from query import query_ephemeris
    # query_ephemeris(BODY_XYZ)

    # --- Use ephemeris data to estimate the covariance matrices and the other parameters
    # from statistics import (measurement_covariance_estimation, time_dependence_estimation,
    #                         process_covariance_estimation)
    # from parse import parse_ephemeris
    # dir_var, vv_var = measurement_covariance_estimation(body)
    # print(f'{dir_var=:.1E}, {vv_var=:.1E}')
    # dt_exp = time_dependence_estimation(body)
    # dt_exp = 1.50
    # print(f'{dt_exp=:.2f}')
    # mm_var, a_var, e_var, i_var, om_var, w_var, n_var = process_covariance_estimation(body, dt_exp)
    # print(f'{mm_var=:.1E}, {a_var=:.1E}, {e_var=:.1E}, {i_var=:.1E}, {om_var=:.1E}, {w_var=:.1E}, {n_var=:.1E}')
    # eph0 = parse_ephemeris(body).__iter__().__next__()
    # mm0, om0 = eph0.mean_anomaly, eph0.ascending_longitude
    # print(f'{mm0=:.2f}, {om0=:.2f}')

    # --- Compute five-number summary
    # (n1, n2, n3, n4, n5), (u1, u2, u3, u4, u5) = five_number_summary(body)
    # print(f'naive = [{n1:.2E}, {n2:.2E}, {n3:.2E}, {n4:.2E}, {n5:.2E}], '
    #       f'ukf = [{u1:.2E}, {u2:.2E}, {u3:.2E}, {u4:.2E}, {u5:.2E}]')

    # --- Visualize error graph and 3D data for new body
    # from visualize import plot_errors, plot_3d
    # plot_errors(body, use_time=False)
    # plot_3d(body)

    # --- Visualize existing body
    from visualize import plot_3d_bennu, plot_errors_bennu
    plot_3d_bennu()
    plot_errors_bennu()


if __name__ == '__main__':
    main()
