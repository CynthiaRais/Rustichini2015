from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib.ticker import IndexLocator, FormatStrFormatter
from matplotlib.markers import MarkerStyle

import numpy as np


def _prepare_plot(x_label='offer A', y_label='offer B', z_label=None, title='', z_ticks=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.set_autoscaley_on(False)
    ax.invert_xaxis()

    ax.grid(linestyle='dashed')
    ax.grid(which='major', alpha=0.1)
    ax.view_init(azim=-35, elev=5)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(IndexLocator(0.5, 0))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # White background
    for item in (ax.xaxis, ax.yaxis, ax.zaxis):
        item.set_pane_color((1.0, 1.0, 1.0, 1.0))
        item.label.set_size(7)

    for item in (ax.get_xticklabels(), ax.get_yticklabels(), ax.get_zticklabels()):
        plt.setp(item, fontsize=7)

    # Ticks
    xy_ticks = [0, 10, 20]
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.FixedLocator(xy_ticks))
        axis.set_major_formatter(ticker.FixedFormatter(['{:d}'.format(tick) for tick in xy_ticks]))

    if z_ticks is not None:
        ax.zaxis.set_major_locator(ticker.FixedLocator(z_ticks))

    # Labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)

    plt.title(title)

    return fig, ax


def tuningcurve(XYZC, show=True, filename_suffix='', **kwargs):
    """
    Tuning curve plotting code for Fig. 4.[B, F, J], Fig. 6.[B, F, J], Fig. 10.[C, I].

    :param XYZC:  list of (x, y, z, choice) values, with choice either 'A' or 'B'.
    """

    fig, ax = _prepare_plot(**kwargs)

    XA, YA, ZA = [], [], []
    XB, YB, ZB = [], [], []
    for x, y, z, c in XYZC:
        if c == 'A':
            XA.append(x)
            YA.append(y)
            ZA.append(z)
        else:
            assert c == 'B'
            XB.append(x)
            YB.append(y)
            ZB.append(z)

    # Plot the markers.
    surf = ax.scatter(XA, YA, ZA, marker='D', edgecolor='red',  facecolor=(0,0,0,0), s=50)
    surf = ax.scatter(XB, YB, ZB, marker='o', edgecolor='blue', facecolor=(0,0,0,0), s=70)

    plt.savefig('figures/{}{}.pdf'.format(kwargs['title'], filename_suffix))

    if show:
        plt.show()



def regression_3D(data, show=True, filename_suffix='', **kwargs):
    X , Y, Z, X_reg, Y_reg, Z_reg = data
    fig, ax = _prepare_plot(**kwargs)
    ax.view_init(azim=-35, elev=31)
    ax.set_zlim(0.0, 100.0)

    ax.plot_surface(X_reg, Y_reg, Z_reg, cmap=cm.jet, linewidth=0, antialiased=True)
    ax.scatter(X, Y, Z, marker='.', edgecolor='grey',  facecolor=(0,0,0,0), s=20)

    plt.savefig('figures/{}{}.pdf'.format(kwargs['title'], filename_suffix))
    if show:
        plt.show()
