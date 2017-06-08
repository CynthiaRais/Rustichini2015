import numpy as np

import bokeh
import bokeh.plotting as bpl
from bokeh.models import FixedTicker, FuncTickFormatter

from . import utils_bokeh
from . import graphs3d


A_color = '#c5392b'
B_color = '#2e3abf'
grey_high   = '#333333'
grey_medium = '#8c8c8c'
grey_low    = '#cccccc'


class Graph:

    def __init__(self, analysis):
        self.x_axis = 1000 * np.arange(-0.5, 1.0, analysis.history.model.dt)
        self.x_range = np.min(self.x_axis), np.ceil(np.max(self.x_axis))
        bpl.output_notebook(hide_banner=True)

    def fix_x_ticks(self, fig):
        fig.xaxis[0].ticker = FixedTicker(ticks=[-500, 0, 500, 1000])

    def firing_time_ov(self, mean_firing_rates_ov):
        figure_4A = bpl.figure(title="Figure 4A", plot_width=300, plot_height=300, tools=(),
                               x_range=self.x_range, y_range=(0, 7.0))
        utils_bokeh.tweak_fig(figure_4A)
        self.fix_x_ticks(figure_4A)
        figure_4A.yaxis[0].ticker = FixedTicker(ticks=[0, 2, 4, 6])

        figure_4A.multi_line([self.x_axis, self.x_axis, self.x_axis],
                             mean_firing_rates_ov,
                             color=[grey_low, grey_medium, grey_high], line_width=4)
        bpl.show(figure_4A)

    def tuning_curve_ovb(self, tuning_ovb, title='Figure 4B'):
        graphs3d.tuningcurve(tuning_ovb, x_label='offer A', y_label='offer B', title=title)

    def tuning_curve_cj(self, tuning_cjb):
        graphs3d.tuningcurve(tuning_cjb, x_label='offer A', y_label='offer B', title='Figure 4F')

    def tuning_curve_cv(self, tuning_cv):
        graphs3d.tuningcurve(tuning_cv, x_label='offer A', y_label='offer B', title='Figure 4J')


    def firing_specific_set_ovb(self, offers, ovb_choices, percents_B):
        """Figure 4C"""
        self.specific_set_graphs(offers, ovb_choices, percents_B, (0, 5), 'Figure 4C')


    def firing_specific_set_cjb(self, offers, cjb_choices, percents_B):
        """Figure 4G"""
        self.specific_set_graphs(offers, cjb_choices, percents_B, (0, 16), 'Figure 4G')


    def firing_specific_set_cv(self, offers, cv_choices, percents_B):
        """Figure 4K"""
        self.specific_set_graphs(offers, cv_choices, percents_B, (10, 17), 'Figure 4K')


    def specific_set_graphs(self, offers, firing_rate, percents_B, y_range, title):
        """Figure 4C, 4G, 4K"""
        figure = bpl.figure(title=title, plot_width=300, plot_height=300, tools=(),
                            y_range=(y_range[0] - 0.05 * (y_range[1] - y_range[0]), y_range[1]))
        utils_bokeh.tweak_fig(figure)

        figure.xaxis[0].ticker = FixedTicker(ticks=list(range(len(offers))))
        figure.xaxis.formatter = FuncTickFormatter(code="""
            var labels = {};
            return labels[tick];
        """.format({i: '{}B:{}A'.format(x_B, x_A) for i, (x_A, x_B) in enumerate(offers)}))
        figure.xaxis.major_label_orientation = np.pi / 2

        y_min, y_max = y_range
        K = 0.94 * (y_max - y_min) + y_min
        xs = [offers.index(key) for key in percents_B.keys()]
        ys = [y * 0.94 * (y_max - y_min) + y_min for y in percents_B.values()]

        figure.line(x=(min(xs), max(xs)), y=(K, K), color="black", line_dash='dashed')
        figure.circle(x=xs, y=ys, color="black", size=15)

        r_A, r_B = firing_rate
        xs_A = [offers.index(key) for key in r_A.keys()]
        xs_B = [offers.index(key) for key in r_B.keys()]

        figure.diamond(x=xs_A, y=list(r_A.values()), size=15, color=A_color, alpha=0.75)
        figure.circle( x=xs_B, y=list(r_B.values()), size=10, color=B_color, alpha=0.75)

        bpl.show(figure)


    def firing_offer_B(self, tuning_ovb):
        xs_diamonds, ys_diamonds = [], []
        xs_circles,  ys_circles  = [], []
        for (x_A, x_B, r_ovb, choice) in tuning_ovb:
            if choice == 'A':
                xs_diamonds.append(x_B)
                ys_diamonds.append(r_ovb)
            else:
                xs_circles.append(x_B)
                ys_circles.append(r_ovb)

        figure_4D = bpl.figure(title="Figure 4D", plot_width=300, plot_height=300,
                               y_range=[0, 5], tools=())
        utils_bokeh.tweak_fig(figure_4D)

        figure_4D.diamond(xs_diamonds, ys_diamonds, size=15, fill_color=None, line_color=A_color, line_alpha=0.5)
        figure_4D.circle(xs_circles, ys_circles, size=10, fill_color=None, line_color=B_color, line_alpha=0.5)

        #figure_4_D.annulus(x=range(21), y=firing_D, color="purple", inner_radius=0.1, outer_radius=0.2)
        bpl.show(figure_4D)


    def firing_time_cjb(self, mean_firing_rates_cjb):
        figure_4E = bpl.figure(title="Figure 4E", plot_width=300, plot_height=300, tools=(),
                               x_range=self.x_range, y_range=(0, 25.0))
        utils_bokeh.tweak_fig(figure_4E)
        self.fix_x_ticks(figure_4E)

        figure_4E.multi_line([self.x_axis, self.x_axis],
                             mean_firing_rates_cjb,
                             color=[grey_low, grey_high], line_width=4)
        bpl.show(figure_4E)

    def firing_choice(self, tunnig_cjb):
        """Figure 4H"""
        figure_4H = bpl.figure(title="Figure 4H", plot_width=300, plot_height=300,
                               x_range=[0.75, 2.25], y_range=[0, 18], tools=())

        utils_bokeh.tweak_fig(figure_4H)
        figure_4H.xaxis[0].ticker = FixedTicker(ticks=[1, 2])
        figure_4H.yaxis[0].ticker = FixedTicker(ticks=[0, 5, 10, 15])
        figure_4H.xaxis.formatter = FuncTickFormatter(code="""
            var labels = {};
            return labels[tick];
        """.format({1: 'A chosen', 2: 'B chosen'}))

        y_A = [r_cjb for x_A, x_B, r_cjb, choice in tunnig_cjb if choice == 'A']
        y_B = [r_cjb for x_A, x_B, r_cjb, choice in tunnig_cjb if choice == 'B']
        figure_4H.diamond(x=len(y_A)*[1], y=y_A, size=15, fill_color=None, line_color=A_color, line_alpha=0.5)
        figure_4H.circle (x=len(y_B)*[2], y=y_B, size=10, fill_color=None, line_color=B_color, line_alpha=0.501)

        bpl.show(figure_4H)


    def firing_time_cv(self, mean_firing_rates_cv):
        figure_4I = bpl.figure(title="Figure 4I", plot_width=300, plot_height=300, tools=())
        utils_bokeh.tweak_fig(figure_4I)
        self.fix_x_ticks(figure_4I)

        figure_4I.multi_line([self.x_axis, self.x_axis, self.x_axis],
                              mean_firing_rates_cv,
                              color=[grey_low, grey_medium, grey_high], line_width=4)

        bpl.show(figure_4I)


    def firing_chosen_value(self, mean_firing_rate_chosen_value):
        xs_A, ys_A, xs_B, ys_B = mean_firing_rate_chosen_value

        figure_4L = bpl.figure(title="Figure 4 L", plot_width=300, plot_height=300, tools=())
        utils_bokeh.tweak_fig(figure_4L)

        figure_4L.diamond(x=xs_A, y=ys_A, color=A_color, size=15, fill_color=None, line_color=A_color, line_alpha=0.5)
        figure_4L.circle( x=xs_B, y=ys_B, color=B_color, size=10, fill_color=None, line_color=B_color, line_alpha=0.5)
        bpl.show(figure_4L)


    ###

    def logistic_regression(self, X):
        graphs3d.tuningcurve(X,)

    def cja_cjb(self):
        figure_9A = bpl.figure(title="Figure 9 A", plot_width=700, plot_height=700)


    def test(self, firing_cj, firing_cv):
        figure_test_cj = bpl.figure(title="Figure test cj a", plot_width=300, plot_height=300)
        figure_test_cj.line(x=np.arange(-1, 1, 0.0005 ), y=firing_cj, color="red")
        figure_test_cv = bpl.figure(title="Figure test cj a", plot_width=300, plot_height=300)
        figure_test_cv.line(x=np.arange(-1, 1, 0.0005), y=firing_cv, color="red")
        bpl.show(figure_test_cj)
        bpl.show(figure_test_cv)


        #graphs3d.tuningcurve(self.analysis.tuning_cjb, x_label='offer A', y_label='offer B', title='tuning cj')

"""dans graphs j'ai une fonction tuning-curve qui prend en argument ce dont j'ai besoin pour faire mes tuning curve"""
