import numpy.polynomial.polynomial as poly
import numpy as np
import scipy.signal


class DataAnalysis:

    def __init__(self, model):

        self.model = model

        self.ΔA = self.model.ΔA
        self.ΔB = self.model.ΔB

        self.preprocessing()

    def clear_history(self):
        """Clear trial history.

        To be called after `self.preprocessing` and before pickling the DataAnalysis object,
        to only keep means and choice counts, which are the one used to draw graphs.
        """
        self.model.history.clear()
        self.model.trial_history = None


    def preprocessing(self):
        """Compute useful data arrays"""

        # means of all variables of a trial
        self.means        = {} # indexed by (x_A, x_B)
        self.means_choice = {} # indexed by (x_A, x_B, choice)
        for key, trials in self.model.history.trials.items():
            if len(trials) > 0:
                self.means[key] = {}
                for key2 in trials[0].keys():
                    self.means[key][key2] = np.mean([trial[key2] for trial in trials],
                                                    axis=0, dtype=np.float32)
            else:
                self.means[key] = {}
        for key, trials in self.model.history.trials_choice.items():
            if len(trials) > 0:
                self.means_choice[key] = {}
                for key2 in trials[0].keys():
                    self.means_choice[key][key2] = np.mean([trial[key2] for trial in trials],
                                                           axis=0, dtype=np.float32)
            else:
                self.means_choice[key] = {}

        # number of choices
        self.choices = {}
        choices_A, choices_B = {}, {}
        for (x_A, x_B, choice), trials in self.model.history.trials_choice.items():
            {'A': choices_A, 'B': choices_B}[choice][(x_A, x_B)] = len(trials)
        for key in choices_A.keys():
            self.choices[key] = choices_A[key], choices_B[key]


    def mean_window(self, y):
        return scipy.signal.savgol_filter(y, 11, 1)

    def step_range(self, time_range):
        """Transform a time_range (in seconds) into a step_range.

        The time range is considered relative to the offer time.
        """
        return (round((time_range[0] + self.model.t_offer)/self.model.dt),
                round((time_range[1] + self.model.t_offer)/self.model.dt)+1)


    def means_chosen_choice(self, key, time_window=(-0.5, 1.0)):
        """Return the mean of a variable, according to which choice was made.

        :param key:          name of the variable to consider (e.g. 'r_2', 'r_I')
        :param time_window:  time window in seconds from which to compute the mean (e.g. (0, 0.5)).
                             Times are relative to the offer time.
s
        Used in Figure 4E, 4H, 6A, 6E, 6I
        """
        step_range = self.step_range(time_window)
        chosen_means = {'A': [], 'B': []} # A chosen, B chosen

        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0:
                chosen_means[choice].append(means[key][step_range[0]:step_range[1]])

        return (self.mean_window(np.mean(chosen_means['A'], axis=0)),
                self.mean_window(np.mean(chosen_means['B'], axis=0)))


    def means_lowmedhigh_AB(self, key, time_window=(-0.5, 1.0)):
        """Return the mean of a variable, classified in 3 classes: low, medium and high,
        according to the tertiles of quantity of A or B, considered relative to the choice made.

        :param key:          name of the variable to consider (e.g. 'r_2', 'r_I')
        :param time_window:  time window in seconds from which to compute the mean (e.g. (0.0, 0.5)).
                             Times are relative to the offer time.
        """
        step_range = self.step_range(time_window)
        means_lmh = ([], [], []) # low, medium, high

        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0:
                if choice == 'A':
                    q = int(3 * x_A / (self.ΔA + 1))
                    means_lmh[q].append(means[key][1000:4000])
                if choice == 'B':
                    q = int(3 * x_B / (self.ΔB + 1))
                    means_lmh[q].append(means[key][1000:4000])

        return [self.mean_window(np.mean(m, axis=0)) for m in means_lmh]


    def means_lowmedhigh_B(self, key, time_window=(-0.5, 1.0)):
        """Return the mean of a variable, classified in 3 classes: low, medium and high,
        according to the tertiles of quantity of B, irrespective of the choice made.

        :param key:          name of the variable to consider (e.g. 'r_2', 'r_I')
        :param time_window:  time window in seconds from which to compute the mean (e.g. (1.0, 1.5))
        """
        step_range = self.step_range(time_window)
        means_lmh = ([], [], []) # low, medium, high

        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0:
                q = int(3 * x_B / (self.ΔB + 1))
                means_lmh[q].append(means[key][1000:4000])

        return [self.mean_window(np.mean(m, axis=0)) for m in means_lmh]


    def tuning_curve(self, key, time_window):
        """Return the tuning curve of a variable accross a specific time window.

        :param key:          name of the variable to consider (e.g. 'r_2', 'r_I')
        :param time_window:  time window in seconds from which to compute the mean (e.g. (1.0, 1.5))
        """
        step_range = self.step_range(time_window)
        tuning_data = []

        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(means) > 0 and (x_A, x_B) != (0, 0):
                m = np.mean(means[key][step_range[0]:step_range[1]])
                tuning_data.append((x_A, x_B, m, choice))

        return tuning_data


    def means_chosen_value(self, key, time_window):
        """Get firing rate in function of chosen value (Fig. 4L)"""
        xs_A, ys_A, xs_B, ys_B = [], [], [], []

        for x_A, x_B, r_cv, choice in self.tuning_curve(key, time_window):
            if choice == 'A':
                xs_A.append(x_A * self.model.δ_J_stim['1'])
                ys_A.append(r_cv)
            else:
                xs_B.append(x_B * self.model.δ_J_stim['2'])
                ys_B.append(r_cv)

        return xs_A, ys_A, xs_B, ys_B


    def percents(self, choice, x_offers):
        """Determination of percents of choice B depending on quantity of each juice. (Figure 4C)"""
        percents = {}
        for x_A, x_B in x_offers:
            n_A, n_B = self.choices[(x_A, x_B)]
            if choice == 'A':
                percents[(x_A, x_B)] = n_A/max(1, n_A + n_B)
            else:
                percents[(x_A, x_B)] = n_B/max(1, n_A + n_B)

        return percents


    def means_offers(self, key, offers, time_window):
        """Average of the firing rate of OVB cells for certain offers between 0.0s and 0.5s
        after the offer. (Figure 4C)"""

        step_range = self.step_range(time_window)

        mean_firing_A, mean_firing_B = {}, {}
        for x_A, x_B in offers:
            if len(self.means_choice[(x_A, x_B, 'A')]) > 0:
                mean_firing_A[(x_A, x_B)] = np.mean(self.means_choice[(x_A, x_B, 'A')][key][step_range[0]:step_range[1]])
            if len(self.means_choice[(x_A, x_B, 'B')]) > 0:
                mean_firing_B[(x_A, x_B)] = np.mean(self.means_choice[(x_A, x_B, 'B')][key][step_range[0]:step_range[1]])

        return mean_firing_A, mean_firing_B
