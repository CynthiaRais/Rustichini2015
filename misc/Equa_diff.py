import numpy
import math
import bokeh
import bokeh.plotting

a = 1
x = 1
Y = []
Y2 = []
dt = 0.001
T = numpy.arange(0, 5, dt)


for T_i in T:
    y = numpy.exp(a * T_i)
    Y.append(y)
    Y2.append(x)
    x += a * x * dt

sum_square = sum(((Y[i] - Y2[i]) ** 2 for i in range(len(Y))))
print (sum_square)
difference = math.sqrt(sum_square)
print(difference)

bokeh.plotting.output_notebook()
p = bokeh.plotting.figure(title="Exp-Approx", plot_width=300, plot_height=300)
p.multi_line([T , T],[Y,Y2] , color=['blue','red'])

bokeh.plotting.show(p)
