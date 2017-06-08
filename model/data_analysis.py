import numpy.polynomial.polynomial as poly
import numpy as np


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


    def mean_firing_rate_ovb(self):
        """Average of firing rate of OVB cells depending on the quantity of juice B: low, medium, high (fig.4)"""
        range_k = range(1000, 4000)
        ovb_rates = ([[] for _ in range_k], # low
                     [[] for _ in range_k], # medium
                     [[] for _ in range_k]) # high

        for i in range(0, self.ΔA + 1):
            for j in range(0, self.ΔB + 1):
                q = int(3 * j / (self.ΔB+1))
                if len(self.means[(i, j)]) > 0:
                    for k in range_k:
                        ovb_rates[q][k-1000].append(self.means[(i, j)]['r_ovb'][k])

        for ovb_q in ovb_rates:
            for k, values in enumerate(ovb_q):
                ovb_q[k] = np.mean(values)

        return ovb_rates


    def tuning_curve_ovb(self):
        """Tuning curve for ovb cells"""
        tuning_ov = []
        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0 and (x_A, x_B) != (0, 0):
                m = np.mean(means['r_ovb'][2000: 3001])
                tuning_ov.append((x_A, x_B, m, choice))

        return tuning_ov
