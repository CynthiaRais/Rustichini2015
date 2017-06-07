import numpy as np

import bokeh
import bokeh.plotting as bpl
from bokeh.models import FixedTicker

from . import graphs


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

        X2_axis = ["0B: 1A", "1B: 20A", "1B: 16A", "1B: 12A", "1B: 8A", "1B: 4A", "4B: 1A", "8B: 1A", "12B: 1A",
               "16B: 1A", "20B: 1A", "1B: 0A"]
        bpl.output_notebook(hide_banner=True)

    def fix_x_ticks(self, fig):
        fig.xaxis[0].ticker = FixedTicker(ticks=[-500, 0, 1000])

    def firing_time_ov(self, mean_firing_rates_ov):
        figure_4A = bpl.figure(title="Figure 4A", plot_width=300, plot_height=300, tools=(),
                               x_range=self.x_range, y_range=(0, 7.0))
        graphs.tweak_fig(figure_4A)
        self.fix_x_ticks(figure_4A)
        figure_4A.yaxis[0].ticker = FixedTicker(ticks=[0, 2, 4, 6])

        figure_4A.multi_line([self.x_axis, self.x_axis, self.x_axis],
                             mean_firing_rates_ov,
                             color=['#cccccc', "#8c8c8c", "#333333"], line_width=4)
        bpl.show(figure_4A)

    def tuning_curve_ov(self, tuning_ov):
        graphs.tuningcurve(tuning_ov, x_label='offer A', y_label='offer B', title='tuning ov')


    def firing_specific_set_ov(self, ov_choice, pourcentage_choice_B):
        figure_4_C = bpl.figure(title="Figure 4 C", plot_width=300, plot_height=300)
        figure_4_C.diamond(x=range(0, 6), y=ov_choice[0][:6], color='red', size=10)
        figure_4_C.circle(x=range(6, 12), y=ov_choice[1][6:], color="blue", size=10)
        figure_4_C.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bpl.show(figure_4_C)


    def firing_offer_B(self, firing_D):
        print("firing_D", len(firing_D))
        figure_4_D = bpl.figure(title="Figure 4 D", plot_width=300, plot_height=300)
        figure_4_D.annulus(x=range(21), y=firing_D, color="purple", inner_radius=0.1, outer_radius=0.2)
        bpl.show(figure_4_D)


    def firing_time_cjb(self, Y):
        figure_4_E = bpl.figure(title="Figure 4 E", plot_width=300, plot_height=300)
        figure_4_E.multi_line([self.X_axis, self.X_axis], Y,
                          color=['red', "blue"])
        bpl.show(figure_4_E)


    def tuning_curve_cj(self, tuning_cjb):
        graphs.tuningcurve(tuning_cjb, x_label='offer A', y_label='offer B', title='tuning cj')


    def firing_specific_set_cjb(self, cj_choice, pourcentage_choice_B):
        figure_4_G = bpl.figure(title="Figure 4 G", plot_width=300, plot_height=300)
        figure_4_G.diamond(x=range(0, 6), y=cj_choice[0][:6], color='red', size=10)
        figure_4_G.circle(x=range(6, 12), y=cj_choice[1][6:], color="blue", size=10)
        figure_4_G.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bpl.show(figure_4_G)


    def firing_choice(self, firing_H):
        Y = firing_H
        figure_4_H = bpl.figure(title="Figure 4 H", plot_width=300, plot_height=300)
        figure_4_H.diamond(x=[1 for i in range(len(Y[0]))], y=Y[0], color="red")
        figure_4_H.circle(x=[2 for i in range(len(Y[1]))], y=Y[1], color="blue")
        bpl.show(figure_4_H)


    def firing_time_cv(self, Y):
        figure_4_I = bpl.figure(title="Figure 4 I", plot_width=300, plot_height=300)
        figure_4_I.multi_line([self.X_axis, self.X_axis, self.X_axis],
                              Y,
                              color=['red', "green", "blue"])
        bpl.show(figure_4_I)


    def tuning_curve_cv(self, tuning_cv):
        graphs.tuningcurve(tuning_cv, x_label='offer A', y_label='offer B', title='tuning cv')


    def firing_specific_set_cv(self, cv_choice, pourcentage_choice_B):
        figure_4_K = bpl.figure(title="Figure 4 K", plot_width=300, plot_height=300)
        figure_4_K.diamond(x=range(0, 6), y=cv_choice[0][:6], color='red', size=10)
        figure_4_K.circle(x=range(6, 12), y=cv_choice[1][6:], color="blue", size=10)
        figure_4_K.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bpl.show(figure_4_K)

    def firing_chosen_value(self, Y):
        figure_4_L = bpl.figure(title="Figure 4 L", plot_width=300, plot_height=300)
        figure_4_L.diamond(x=Y[0], y=Y[1], color="red", size=10)
        figure_4_L.circle(x=Y[2], y=Y[3], color="blue", size=10)
        bpl.show(figure_4_L)

    def logistic_regression(self, X):
        graphs.tuningcurve(X,)

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

        #graphs.tuningcurve(self.analysis.tuning_cjb, x_label='offer A', y_label='offer B', title='tuning cj')

"""dans graphs j'ai une fonction tuning-curve qui prend en argument ce dont j'ai besoin pour faire mes tuning curve"""
