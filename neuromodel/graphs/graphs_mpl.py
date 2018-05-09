"""3D graphs using matplotlib"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib.ticker import IndexLocator, FormatStrFormatter
from matplotlib.markers import MarkerStyle

import numpy as np

A_color = '#bd5151' # 189, 81, 81  fig4: '#c5392b'
B_color = '#575aa3' #  87, 90,163  fig4: '#2e3abf'


def _prepare_plot(x_label='offer A', y_label='offer B', title='',
                  z_label=None, z_ticks=None, ΔA=(0, 20), ΔB=(0, 20)):

    fig = plt.figure(figsize=(9, 10))
    ax = fig.gca(projection='3d')
    ax.set_xlim(ΔA)
    ax.set_ylim(ΔB)
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
    x_ticks = [5*i for i in range(int(ΔA[1]/5)+1)]
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['{:d}'.format(tick) for tick in x_ticks]))

    y_ticks = [5*i for i in range(int(ΔB[1]/5)+1)]
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(['{:d}'.format(tick) for tick in y_ticks]))

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


def tuningcurve(XYZC, show=True, model_desc='', fig=None, **kwargs):
    """
    Tuning curve plotting code for Fig. 4.[B, F, J], Fig. 6.[B, F, J], Fig. 10.[C, I].

    :param XYZC:  list of (x, y, z, choice) values, with choice either 'A' or 'B'.
    """

    if fig is None:
        fig, ax = _prepare_plot(**kwargs)
    else:
        ax = fig.gca(projection='3d')

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
    surf = ax.scatter(XA, YA, ZA, marker='D', edgecolor=A_color, linewidths=2,
                      facecolor=(0,0,0,0), s=100)
    surf = ax.scatter(XB, YB, ZB, marker='o', edgecolor=B_color, linewidths=2,
                      facecolor=(0,0,0,0), s=140)

    filepath = 'pdfs/{}{}.pdf'.format(kwargs['title'].replace(' ', '_'), model_desc)
    print('saving {}'.format(filepath))
    plt.savefig(filepath)

    if show:
        plt.show()

    return fig


def regression_3D(data, show=True, model_desc='', point_color='grey',
                  fig=None, azim=-35, elev=5, marker='o', **kwargs):
    X , Y, Z, X_reg, Y_reg, Z_reg = data

    if fig is None:
        fig, ax = _prepare_plot(z_ticks=[0, 25, 50, 75, 100], **kwargs)
        ax.view_init(azim=azim, elev=elev)
    else:
        ax = fig.axes[0] #fig.gca(projection='3d')

    ax.set_zlim(0.0, 100.0)

    ax.plot_surface(X_reg, Y_reg, Z_reg, cmap=cm.jet, alpha=0.5, linewidth=0, antialiased=True)
    ax.scatter(X, Y, Z, marker=marker, edgecolor=point_color, facecolor=point_color, s=20)

    plt.savefig('pdfs/{}{}.pdf'.format(kwargs['title'].replace(' ', '_'), model_desc))
    if show:
        plt.show()

    return fig
