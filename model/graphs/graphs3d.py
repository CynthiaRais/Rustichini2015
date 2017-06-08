from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import IndexLocator, FormatStrFormatter
from matplotlib.markers import MarkerStyle

import numpy as np

def tuningcurve(XYZC, show=True, x_label=None, y_label=None, z_label=None, title=''):
    """
    Tuning curve plotting code for Fig. 4.[B, F, J], Fig. 6.[B, F, J], Fig. 10.[C, I].

    :param XYZC:  list of (x, y, z, choice) values, with choice either 'A' or 'B'.
    """

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.gca().invert_xaxis()

    # Plot the markers.
    surf = ax.scatter(XA, YA, ZA, marker='D', edgecolor='red',  facecolor=(0,0,0,0), s=50)
    surf = ax.scatter(XB, YB, ZB, marker='o', edgecolor='blue', facecolor=(0,0,0,0), s=70)

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

    # Labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)

    plt.title(title)
    plt.savefig('tuningcurve.pdf')
    plt.show()
