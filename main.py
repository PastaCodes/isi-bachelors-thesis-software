from bodies import *


def main() -> None:
    from visualize import plot_3d, plot_errors
    # plot_3d(BODY_2015JJ, compute_from=10)
    plot_errors(BODY_2015JJ, compute_from=10, to=250, use_time=False)


if __name__ == '__main__':
    main()
