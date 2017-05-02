import dotdot
import graphs

XYZC = []
for x in range(20):
    for y in range(20):
        z, c = y/4, 'A' if x > 10 else 'B'
        XYZC.append((x, y, z, c))

graphs.tuningcurve(XYZC, x_label='offer A', y_label='offer B')
