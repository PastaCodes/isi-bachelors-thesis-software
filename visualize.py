import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import scipy.optimize as spo
from matplotlib.widgets import Slider

from main import MinorPlanet, OBJ_BENNU, output_transform, earth_sun_distance
from parse import load_ephemeris


def visualize_phase_equation(obj: MinorPlanet) -> None:
    obss = obj.load_observations()

    from main import elongation_from_observation, earth_sun_distance
    from photometry import VISUAL_CORRECTION, phi

    def params(i: int) -> tuple[float, float, float, float, float]:
        obs = obss[i]
        slope = obs.obj.slope
        visual_magnitude = obs.magnitude + VISUAL_CORRECTION[obs.band]
        elongation = elongation_from_observation(obs)
        es_dist = earth_sun_distance(obs.obstime)
        elong_sin = np.sin(elongation)
        mag_diff = visual_magnitude - obs.obj.absolute_magnitude
        return slope, es_dist, elongation, elong_sin, mag_diff

    i0 = 0
    g, r, th, sin_th, md = params(i0)

    def f(a: float) -> float:
        sin_a = np.sin(a)
        return (r * r * sin_th * np.sin(a + th) -
                np.pow(10, 0.2 * md) * sin_a * sin_a * np.sqrt(phi(a, g)))
    ff = np.vectorize(f)

    aa = np.arange(0.0, np.pi, 0.01)
    yy = ff(aa)
    a_sol: float = spo.brentq(f, 0, np.pi)

    fig, ax = plt.subplots()
    fig.suptitle(f'Visualizzazione dell\'equazione di fase per ‘{obj.name}’')
    ax.set_xlabel('α')
    ax.set_ylabel('f(α)')
    fig.subplots_adjust(bottom=0.2)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylim(-1.2, 1.2)
    l, = ax.plot(aa, yy, lw=2)
    p, = ax.plot(a_sol, 0, 'ro')
    lbl = ax.annotate(f'α = {a_sol:.6f}', xy=(a_sol + 0.05, 0.05), xycoords='data')

    ax_i = fig.add_axes((0.25, 0.05, 0.65, 0.03))
    i_slider = Slider(ax=ax_i,
                      label='Oss. n.',
                      valmin=0,
                      valmax=(len(obss) - 1),
                      valinit=i0,
                      valstep=1)

    def on_changed(i: float):
        nonlocal g, r, th, sin_th, md, yy, a_sol
        g, r, th, sin_th, md = params(int(i))
        yy = ff(aa)
        l.set_ydata(yy)
        a_sol = spo.brentq(f, 0, np.pi)
        p.set_xdata([a_sol])
        lbl.set_text(f'α = {a_sol:.6f}')
        lbl.set_x(a_sol + 0.05)
        fig.canvas.draw_idle()

    i_slider.on_changed(on_changed)

    plt.show()


def error_on_observed_values() -> None:
    obss = OBJ_BENNU.load_observations()
    ephs = list(load_ephemeris(OBJ_BENNU, 'data/Bennu_ephemeris.txt').values())

    assert len(ephs) == len(obss)
    r = range(len(obss))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Errori sui valori osservati per ‘Bennu’')

    ax1.set_title('Errore sull\'ascensione retta α')
    ax1.axhline(0, color='g')
    ax1.plot(r, [(obss[i].right_ascension - ephs[i].right_ascension) * np.pi / 12 for i in r], 'r+')

    ax2.set_title('Errore sulla declinazione δ')
    ax2.axhline(0, color='g')
    ax2.plot(r, [(obss[i].declination - ephs[i].declination) * np.pi / 180 for i in r], 'r+')

    ax3.set_title('Errore sulla magnitudine visuale V')
    ax3.axhline(0, color='g')
    ax3.plot(r, [obss[i].visual_magnitude - ephs[i].visual_magnitude for i in r], 'r+')

    plt.show()


def error_on_elong_and_phase() -> None:
    from main import elongation_from_observation
    from photometry import phase_from_magnitude

    obss = OBJ_BENNU.load_observations()
    ephs = list(load_ephemeris(OBJ_BENNU, 'data/Bennu_ephemeris.txt').values())

    assert len(ephs) == len(obss)
    r = range(len(obss))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Errori sui valori intermedi per ‘Bennu’')

    ax1.set_title('Errore sull\'elongazione θ')
    ax1.axhline(0, color='g')
    ax1.plot(r, [elongation_from_observation(obss[i]) - ephs[i].elongation for i in r], 'r+')

    ax2.set_title('Errore sulla fase α')
    ax2.axhline(0, color='g')
    ax2.plot(r, [phase_from_magnitude(obss[i]) - ephs[i].phase for i in r], 'r+')

    plt.show()


def error_on_observed_state() -> None:
    obss = OBJ_BENNU.load_observations()
    ephs = list(load_ephemeris(OBJ_BENNU, 'data/Bennu_ephemeris.txt').values())

    assert len(ephs) == len(obss)
    r = range(len(obss))

    plt.axhline(0, color='g')
    plt.plot(r, [npl.norm(output_transform(obss[i], ephs[i]).position - ephs[i].position) for i in r], 'r+')
    plt.show()


def compare_observed_with_ephemeris() -> None:
    obss = OBJ_BENNU.load_observations()
    ephemeris = load_ephemeris(OBJ_BENNU, 'data/Bennu_ephemeris.txt')
    observed = []
    for obs in obss:
        eph = ephemeris[obs.datetime.to_decimal_year()]
        try:
            observed.append(output_transform(obs, eph))
        except ValueError:
            print('X')
            continue
    # observed = [output_transform(obs) for obs in obss]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    oo = np.array([o.position for o in observed])
    ax.scatter(oo.T[0], oo.T[1], oo.T[2], c='r', depthshade=False)
    ee = np.array([eph.position for eph in ephemeris.values()])
    ax.scatter(ee.T[0], ee.T[1], ee.T[2], c='g', depthshade=False)

    plt.show()
