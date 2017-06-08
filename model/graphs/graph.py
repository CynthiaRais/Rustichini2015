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
    def __init__(self, ΔA=20, ΔB = 20, t_exp =2.0, dt= 0.0005, analysis=None):

        self.ΔA, self.ΔB = ΔA, ΔB
        self.choice_A, self.choice_B = {}, {}
        self.list_choice = ['A', 'B']
        self.t_exp = t_exp
        self.dt = dt
        self.analysis = analysis
        self.ov, self.cjb, self.cv = {}, {}, {}

        self.x_axis = 1000 * np.arange(-0.5, 1.0, self.dt)
        self.x_range = np.min(self.x_axis), np.ceil(np.max(self.x_axis))

        X2_axis = ["0B: 1A", "1B: 20A", "1B: 16A", "1B: 12A", "1B: 8A", "1B: 4A", "4B: 1A",
                   "8B: 1A", "12B: 1A", "16B: 1A", "20B: 1A", "1B: 0A"]
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


    def firing_specific_set_ov(self, ov_choice, pourcentage_choice_B):
        figure_4_C = bpl.figure(title="Figure 4 C", plot_width=300, plot_height=300)
        figure_4_C.diamond(x=range(0, 6), y=ov_choice[0][:6], color='red', size=10)
        figure_4_C.circle(x=range(6, 12), y=ov_choice[1][6:], color="blue", size=10)
        figure_4_C.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bpl.show(figure_4_C)


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


        # figure_4E = bpl.figure(title="Figure 4E", plot_width=300, plot_height=300)
        # figure_4E.multi_line([self.X_axis, self.X_axis], Y,
        #                   color=['red', "blue"])
        # bpl.show(figure_4E)


    def tuning_curve_cj(self, tuning_cjb):
        graphs3d.tuningcurve(tuning_cjb, x_label='offer A', y_label='offer B', title='Figure 4F')


    def firing_specific_set_cjb(self, cj_choice, pourcentage_choice_B):
        figure_4_G = bpl.figure(title="Figure 4 G", plot_width=300, plot_height=300)
        figure_4_G.diamond(x=range(0, 6), y=cj_choice[0][:6], color='red', size=10)
        figure_4_G.circle(x=range(6, 12), y=cj_choice[1][6:], color="blue", size=10)
        figure_4_G.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bpl.show(figure_4_G)


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


    def tuning_curve_cv(self, tuning_cv):
        graphs3d.tuningcurve(tuning_cv, x_label='offer A', y_label='offer B', title='Figure 4J')


    def firing_specific_set_cv(self, cv_choice, pourcentage_choice_B):
        figure_4_K = bpl.figure(title="Figure 4 K", plot_width=300, plot_height=300)
        figure_4_K.diamond(x=range(0, 6), y=cv_choice[0][:6], color='red', size=10)
        figure_4_K.circle(x=range(6, 12), y=cv_choice[1][6:], color="blue", size=10)
        figure_4_K.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bpl.show(figure_4_K)

    def firing_chosen_value(self, mean_firing_rate_chosen_value):
        xs_A, ys_A, xs_B, ys_B = mean_firing_rate_chosen_value

        figure_4L = bpl.figure(title="Figure 4 L", plot_width=300, plot_height=300, tools=())
        utils_bokeh.tweak_fig(figure_4L)

        figure_4L.diamond(x=xs_A, y=ys_A, color=A_color, size=15, fill_color=None, line_color=A_color, line_alpha=0.5)
        figure_4L.circle( x=xs_B, y=ys_B, color=B_color, size=10, fill_color=None, line_color=B_color, line_alpha=0.5)
        bpl.show(figure_4L)

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

    def figure_4(self):

        figure_test_cjb = bpl.figure(title="Figure test cj b", plot_width=300, plot_height=300)
        figure_test_ns = bpl.figure(title="Figure test ns", plot_width=300, plot_height=300)
        figure_test_cv = bpl.figure(title="Figure test cv", plot_width=300, plot_height=300)

        #figure_test_cja.multi_line([self.X_axis, self.X_axis, self.X_axis], self.analysis.test_cja, color=["red", "blue", "green"])
        figure_test_cjb.multi_line([self.X_axis, self.X_axis, self.X_axis], self.analysis.test_cjb, color=["red", "blue", "green"])
        figure_test_ns.multi_line([self.X_axis, self.X_axis,self. X_axis], self.analysis.test_ns, color=["red", "blue", "green"])
        figure_test_cv.multi_line([self.X_axis, self.X_axis, self.X_axis], self.analysis.test_cv, color=["red", "blue", "green"])


        #bpl.show(figure_4_A)
        #bpl.show(figure_4_C)
        #bpl.show(figure_4_D)

        #bpl.show(figure_4_E)
        #bpl.show(figure_4_G)
        #bpl.show(figure_4_H)

        #bpl.show(figure_4_I)
        #bpl.show(figure_4_K)
        #bpl.show(figure_4_L)

        #bpl.show(figure_test_cja)
        bpl.show(figure_test_cjb)
        bpl.show(figure_test_ns)
        bpl.show(figure_test_cv)






    def figure_6(self):


        figure_4_E = bpl.figure(title="Figure 4 E", plot_width=700, plot_height=700)
        figure_4_I = bpl.figure(title="Figure 4 I", plot_width=700, plot_height=700)
        figure_4_L = bpl.figure(title="Figure 4 L", plot_width=700, plot_height=700)

        figure_4_E.multi_line([self.X_axis, self.X_axis], [self.analysis.mean_A_chosen_cj, self.analysis.mean_B_chosen_cj],
                              color=['red', "blue"])
        figure_4_I.multi_line([self.X_axis, self.X_axis, self.X_axis],
                              [self.analysis.mean_low_cv, self.analysis.mean_medium_cv, self.analysis.mean_high_cv],
                              color=['red', "green", "blue"])
        figure_4_L.diamond(x=self.analysis.X_A, y=self.analysis.Y_A, color="red", size=10)
        figure_4_L.circle(x=self.analysis.X_B, y=self.analysis.Y_B, color="blue", size=10)

        bpl.show(figure_4_E)
        bpl.show(figure_4_I)
        bpl.show(figure_4_L)

        #graphs3d.tuningcurve(self.analysis.tuning_cjb, x_label='offer A', y_label='offer B', title='tuning cj')

"""dans graphs j'ai une fonction tuning-curve qui prend en argument ce dont j'ai besoin pour faire mes tuning curve"""
