import numpy.polynomial.polynomial as poly
import numpy as np
import scipy.signal


class DataAnalysis:

    def __init__(self, history):

        self.history = history

        self.ΔA = history.model.ΔA
        self.ΔB = history.model.ΔB

        self.preprocessing()

    def preprocessing(self):
        """Compute useful data arrays"""

        # means of all variables of a trial
        self.means        = {} # indexed by (x_A, x_B)
        self.means_choice = {} # indexed by (x_A, x_B, choice)
        for key, trials in self.history.trials.items():
            if len(trials) > 0:
                self.means[key] = {}
                for key2 in trials[0].keys():
                    self.means[key][key2] = np.mean([trial[key2] for trial in trials], axis=0)
            else:
                self.means[key] = {}
        for key, trials in self.history.trials_choice.items():
            if len(trials) > 0:
                self.means_choice[key] = {}
                for key2 in trials[0].keys():
                    self.means_choice[key][key2] = np.mean([trial[key2] for trial in trials], axis=0)
            else:
                self.means_choice[key] = {}

        # number of choices
        self.choices = {}
        choices_A, choices_B = {}, {}
        for (x_A, x_B, choice), trials in self.history.trials_choice.items():
            {'A': choices_A, 'B': choices_B}[choice][(x_A, x_B)] = len(trials)
        for key in choices_A.keys():
            self.choices[key] = choices_A[key], choices_B[key]


    def mean_window(self, y):
        return scipy.signal.savgol_filter(y, 11, 1)


    def mean_firing_rate_ovb(self):
        """Average of firing rate of OVB cells depending on the quantity of juice B: low, medium, high (Fig. 4A)"""
        range_k = range(1000, 4000)
        ovb_rates = ([[] for _ in range_k], # low
                     [[] for _ in range_k], # medium
                     [[] for _ in range_k]) # high

        for x_A in range(0, self.ΔA + 1):
            for x_B in range(0, self.ΔB + 1):
                q = int(3 * x_B / (self.ΔB+1))
                if len(self.means[(x_A, x_B)]) > 0:
                    for k in range_k:
                        ovb_rates[q][k-1000].append(self.means[(x_A, x_B)]['r_ovb'][k])

        for ovb_q in ovb_rates:
            for k, values in enumerate(ovb_q):
                ovb_q[k] = np.mean(values)

        return ovb_rates


    def mean_firing_rate_cjb(self):
        """Figure 4E, 4H"""
        cjb_rates = {'A': [], 'B': []} # A chosen, B chosen

        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0:
                cjb_rates[choice].append(means['r_2'][1000:4000])

        return (self.mean_window(np.mean(cjb_rates['A'], axis=0)),
                self.mean_window(np.mean(cjb_rates['B'], axis=0)))


    def mean_firing_rate_cv(self):
        """Figure 4I"""

        range_k = range(1000, 4000)
        cv_rates = ([], [], []) # high

        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0:
                if choice == 'A':
                    q = int(3 * x_A / (self.ΔA + 1))
                    cv_rates[q].append(means['r_I'][1000:4000])
                if choice == 'B':
                    q = int(3 * x_B / (self.ΔB + 1))
                    cv_rates[q].append(means['r_I'][1000:4000])

        return [self.mean_window(np.mean(rates, axis=0)) for rates in cv_rates]


    def mean_firing_chosen_value(self):
        """Get firing rate in function of chosen value (Fig. 4L)"""
        xs_A, ys_A, xs_B, ys_B = [], [], [], []

        for x_A, x_B, r_cv, choice in self.tuning_curve_cv():
            if choice == 'A':
                xs_A.append(x_A * self.history.model.δ_J_stim['1'])
                ys_A.append(r_cv)
            else:
                xs_B.append(x_B * self.history.model.δ_J_stim['2'])
                ys_B.append(r_cv)

        return xs_A, ys_A, xs_B, ys_B


    def tuning_curve_ovb(self):
        """Tuning curve for OVB cells (Fig. 4B)"""
        tuning_ovb = []
        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0 and (x_A, x_B) != (0, 0):
                m = np.mean(means['r_ovb'][2000:3000])
                tuning_ovb.append((x_A, x_B, m, choice))

        return tuning_ovb


    def tuning_curve_cjb(self):
        """Tuning curve for CJB cells (Fig. 4F)"""
        tuning_cjb = []
        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0 and (x_A, x_B) != (0, 0):
                m = np.mean(means['r_2'][3000:4000])
                tuning_cjb.append((x_A, x_B, m, choice))

        return tuning_cjb


    def tuning_curve_cv(self):
        """Tuning curve for CV cells (Fig. 4J)"""
        tuning_cv = []
        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0 and (x_A, x_B) != (0, 0):
                m = np.mean(means['r_I'][2000:3000])
                tuning_cv.append((x_A, x_B, m, choice))

        return tuning_cv
