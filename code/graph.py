
import bokeh
import bokeh.plotting
import numpy as np
import graphs



class Graph:
    def __init__(self, ΔA=20, ΔB = 20, t_exp =2.0, dt= 0.0005, analysis=None):

        self.ΔA, self.ΔB = ΔA, ΔB
        self.choice_A, self.choice_B = {}, {}
        self.list_choice = ['A', 'B']
        self.t_exp = t_exp
        self.dt = dt
        self.analysis = analysis
        self.ov, self.cjb, self.cv = {}, {}, {}
        self.X_axis = np.arange(0, self.t_exp, self.dt)
        X2_axis = ["0B: 1A", "1B: 20A", "1B: 16A", "1B: 12A", "1B: 8A", "1B: 4A", "4B: 1A", "8B: 1A", "12B: 1A",
               "16B: 1A", "20B: 1A", "1B: 0A"]
        bokeh.plotting.output_notebook()

    def firing_time_ov(self, Y):
        figure_4_A = bokeh.plotting.figure(title="Figure 4 A", plot_width=700, plot_height=700)
        figure_4_A.multi_line([self.X_axis, self.X_axis, self.X_axis],
                              Y,
                              color=['red', "green", "blue"])
        bokeh.plotting.save(figure_4_A, title="figure 4 A 4000_5")

    def tuning_curve_ov(self, tuning_ov):
        graphs.tuningcurve(tuning_ov, x_label='offer A', y_label='offer B', title='tuning ov')

    def firing_specific_set_ov(self, ov_choiceA, ov_choiceB, pourcentage_choice_B):
        figure_4_C = bokeh.plotting.figure(title="Figure 4 C", plot_width=700, plot_height=700)
        figure_4_C.diamond(x=range(0, 6), y=ov_choiceA, color='red', size=10)
        figure_4_C.circle(x=range(6, 12), y=ov_choiceB, color="blue", size=10)
        figure_4_C.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bokeh.plotting.save(figure_4_C, title="figure 4 C 4000_5")

    def firing_offer_B(self, firing_D):
        figure_4_D = bokeh.plotting.figure(title="Figure 4 D", plot_width=700, plot_height=700)
        figure_4_D.annulus(x=range(20), y=firing_D, color="purple", inner_radius=0.2, outer_radius=0.5)
        bokeh.plotting.save(figure_4_D, title="figure 4 D 4000_5")

    def firing_time_cjb(self, Y):
        figure_4_E = bokeh.plotting.figure(title="Figure 4 E", plot_width=700, plot_height=700)
        figure_4_E.multi_line([self.X_axis, self.X_axis], Y,
                          color=['red', "blue"])
        bokeh.plotting.save(figure_4_E, title="figure 4 E 4000_5")

    def tuning_curve_cj(self, tuning_cjb):
        graphs.tuningcurve(tuning_cjb, x_label='offer A', y_label='offer B', title='tuning cj')

    def firing_specific_set_cjb(self, cjb_choiceA, cjb_choiceB, pourcentage_choice_B):
        figure_4_G = bokeh.plotting.figure(title="Figure 4 G", plot_width=700, plot_height=700)
        figure_4_G.diamond(x=range(0, 6), y=cjb_choiceA, color='red', size=10)
        figure_4_G.circle(x=range(6, 12), y=cjb_choiceB, color="blue", size=10)
        figure_4_G.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        bokeh.plotting.save(figure_4_G, title="figure 4 G 4000_5")

    def firing_choice(self, firing_H_A, firing_H_B):
        figure_4_H = bokeh.plotting.figure(title="Figure 4 H", plot_width=700, plot_height=700)
        figure_4_H.diamond(x=[1 for i in range(len(firing_H_A))], y=firing_H_A, color="red")
        figure_4_H.circle(x=[2 for i in range(len(firing_H_B))], y=firing_H_B, color="blue")
        print("for H", len(firing_H_A), len(firing_H_B), len([1 for i in range(len(firing_H_A))]),
              len([2 for i in range(len(firing_H_B))]))
        bokeh.plotting.save(figure_4_H, title="figure 4 H 4000_5")

    def firing_time_cv(self, Y):
        figure_4_I = bokeh.plotting.figure(title="Figure 4 I", plot_width=700, plot_height=700)
        figure_4_I.multi_line([self.X_axis, self.X_axis, self.X_axis],
                              Y,
                              color=['red', "green", "blue"])
        bokeh.plotting.save(figure_4_I, title="figure 4 I 4000_5")

    def tuning_curve_cv(self, tuning_cv):
        graphs.tuningcurve(tuning_cv, x_label='offer A', y_label='offer B', title='tuning cv')

    def firing_specific_set_cv(self, cv_choiceA, cv_choiceB, pourcentage_choice_B):
        figure_4_K = bokeh.plotting.figure(title="Figure 4 K", plot_width=700, plot_height=700)
        figure_4_K.diamond(x=range(0, 6), y=cv_choiceA, color='red', size=10)  # choiceA
        figure_4_K.circle(x=range(6, 12), y=cv_choiceB, color="blue", size=10)  # choiceB
        figure_4_K.circle(x_range=range(12), y=pourcentage_choice_B, color="black", size=10)  # %choiceB
        bokeh.plotting.save(figure_4_K, title="figure 4 K 4000_5")

    def firing_chosen_value(self, X_A, X_B, Y_A, Y_B):
        figure_4_L = bokeh.plotting.figure(title="Figure 4 L", plot_width=700, plot_height=700)
        print("X_A", len(X_A))
        print("Y_A", len(Y_A))
        print("X_B", len(X_B))
        print("Y_B", len(Y_B))
        figure_4_L.diamond(x=X_A, y=Y_A, color="red", size=10)
        figure_4_L.circle(x=X_B, y=Y_B, color="blue", size=10)
        bokeh.plotting.save(figure_4_L, title="figure 4 L 4000_5")

    def cja_cjb(self):
        figure_9A = bokeh.plotting.figure(title="Figure 9 A", plot_width=700, plot_height=700)



    def figure_4(self):
        figure_test_cj_a = bokeh.plotting.figure(title="Figure test cj a", plot_width=700, plot_height=700)
        figure_test_cj_b = bokeh.plotting.figure(title="Figure test cj b", plot_width=700, plot_height=700)
        figure_test_ns = bokeh.plotting.figure(title="Figure test ns", plot_width=700, plot_height=700)
        figure_test_cv = bokeh.plotting.figure(title="Figure test cv", plot_width=700, plot_height=700)

        figure_test_cj_a.multi_line([self.X_axis, self.X_axis, self.X_axis], self.analysis.test_cj_a, color=["red", "blue", "green"])
        figure_test_cj_b.multi_line([self.X_axis, self.X_axis, self.X_axis], self.analysis.test_cj_b, color=["red", "blue", "green"])
        figure_test_ns.multi_line([self.X_axis, self.X_axis,self. X_axis], self.analysis.test_ns, color=["red", "blue", "green"])
        figure_test_cv.multi_line([self.X_axis, self.X_axis, self.X_axis], self.analysis.test_cv, color=["red", "blue", "green"])


        #bokeh.plotting.show(figure_4_A)
        #bokeh.plotting.show(figure_4_C)
        #bokeh.plotting.show(figure_4_D)

        #bokeh.plotting.show(figure_4_E)
        #bokeh.plotting.show(figure_4_G)
        #bokeh.plotting.show(figure_4_H)

        #bokeh.plotting.show(figure_4_I)
        #bokeh.plotting.show(figure_4_K)
        #bokeh.plotting.show(figure_4_L)

        bokeh.plotting.show(figure_test_cj_a)
        bokeh.plotting.show(figure_test_cj_b)
        bokeh.plotting.show(figure_test_ns)
        bokeh.plotting.show(figure_test_cv)






    def figure_6(self):


        figure_4_E = bokeh.plotting.figure(title="Figure 4 E", plot_width=700, plot_height=700)
        figure_4_I = bokeh.plotting.figure(title="Figure 4 I", plot_width=700, plot_height=700)
        figure_4_L = bokeh.plotting.figure(title="Figure 4 L", plot_width=700, plot_height=700)

        figure_4_E.multi_line([self.X_axis, self.X_axis], [self.analysis.mean_A_chosen_cj, self.analysis.mean_B_chosen_cj],
                              color=['red', "blue"])
        figure_4_I.multi_line([self.X_axis, self.X_axis, self.X_axis],
                              [self.analysis.mean_low_cv, self.analysis.mean_medium_cv, self.analysis.mean_high_cv],
                              color=['red', "green", "blue"])
        figure_4_L.diamond(x=self.analysis.X_A, y=self.analysis.Y_A, color="red", size=10)
        figure_4_L.circle(x=self.analysis.X_B, y=self.analysis.Y_B, color="blue", size=10)

        bokeh.plotting.show(figure_4_E)
        bokeh.plotting.show(figure_4_I)
        bokeh.plotting.show(figure_4_L)

        #graphs.tuningcurve(self.analysis.tuning_cjb, x_label='offer A', y_label='offer B', title='tuning cj')

"""dans graphs j'ai une fonction tuning-curve qui prend en argument ce dont j'ai besoin pour faire mes tuning curve"""