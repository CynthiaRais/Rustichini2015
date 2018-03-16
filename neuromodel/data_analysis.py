import numpy.polynomial.polynomial as poly
import numpy as np
import scipy.signal
import scipy.optimize

class DataAnalysis:

    def __init__(self, model, preprocess=True):

        self.model = model

        self.ΔA = self.model.ΔA
        self.ΔB = self.model.ΔB

        if preprocess:
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


    def mean_window(self, y, size=201):
        return scipy.signal.savgol_filter(y, size, 1)

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
                    q = int(3 * x_A / (self.ΔA[1] + 1))
                    means_lmh[q].append(means[key][1000:4000])
                if choice == 'B':
                    q = int(3 * x_B / (self.ΔB[1] + 1))
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
                q = int(3 * x_B / (self.ΔB[1] + 1))
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


    def approx_polynome(self, x, a_0, a_1, a_2, a_3, a_4, a_5):
        """Approximation of the polynome for the regression"""
        x_a, x_b = x
        X = a_0 + a_1 * x_a + a_2 * x_b + a_3 * x_a * x_a + a_4 * x_b * x_b + a_5 * x_a * x_b
        return 1 / (1 + np.exp(-X))

    def data_regression(self, dim='3D'):
        X_A, X_B, choice_B = [], [], []
        for (x_a, x_b), (n_a, n_b) in sorted(self.choices.items()):
            if x_a != 0 or x_b != 0:
                X_A.append(x_a)
                X_B.append(x_b)
                choice_B.append(n_b / (n_a + n_b))
        a_opt, a_cov = scipy.optimize.curve_fit(self.approx_polynome, [X_A, X_B], choice_B, bounds=((-20,) * 6, (20,) * 6))

        # computing the regressed model over all possible quantities by 0.5 increments.
        X_A_reg = np.arange(0, 20.5, 0.5)
        X_B_reg = np.arange(0, 20.5, 0.5)
        X_A_reg, X_B_reg = np.meshgrid(X_A_reg, X_B_reg)
        choice_B_reg = 100 * self.approx_polynome((X_A_reg, X_B_reg), *a_opt)

        if dim == '3D':
            return X_A, X_B, 100*np.array(choice_B), X_A_reg, X_B_reg, choice_B_reg
        elif dim == '2D':
            return 100 * self.approx_polynome((X_B_reg, X_A_reg), *a_opt)
        else:
            return ValueError

    def easy_split_fig8(self, key, A_offers=range(10)):
        # FIXME: verify experimentally which trials are split and which are not.
        """Return mean firing rates, divided between easy and split offers, for Figure 8B.

        'Split' offers are offers where the model sometimes chooses A, sometimes B. 'Easy' offers
        are the one where the model always make the same decision. The denomination can be made
        experimentally, or theoretically. Here, the authors have decided to consider that offers
        where the value of A is between 1 and 10 may be split, but not the others.

        From the article: "For each quantity of A that induced some split decisions (i.e., offer
        value A 1...10), trials in which the network chose juice A were divided into easy and split"

        :param key:          name of the variable to consider (e.g. 'r_2', 'r_I')

        Used in Figure 8B.
        """
        A_means = {'easy': [], 'split':[]}
        step_range = self.step_range((-0.5, 1.0))

        for A_offer in A_offers:
            firing_rates = {'easy': [], 'split': []}
            for (x_A, x_B, choice), means in self.means_choice.items():
                if x_A == A_offer and choice == 'A' and len(means) > 0:
                    if len(self.means_choice[(x_A, x_B, 'B')]) > 0:  # B was also chosen sometimes
                        for _ in range(len(self.means_choice[(x_A, x_B, 'A')])):
                            firing_rates['split'].append(means[key][step_range[0]:step_range[1]])
                    else:
                        for _ in range(len(self.means_choice[(x_A, x_B, 'A')])):
                            firing_rates['easy'].append(means[key][step_range[0]:step_range[1]])
            A_means['easy'].append(np.mean(firing_rates['easy'], axis=0))
            A_means['split'].append(np.mean(firing_rates['split'], axis=0))

        return (self.mean_window(np.mean(A_means['easy'], axis=0), size=201),
                self.mean_window(np.mean(A_means['split'], axis=0), size=201))

    def choice_hysteresis(self, key='r_2', time_window=(-0.5, 1.0)):
        self.previous = {'split': {'A': [], 'B': []}, 'easy': {'A': [], 'B': []}}
        step_range = self.step_range(time_window)
        for (x_A, x_B, choice), means in self.means_choice.items():
            if len(self.means_choice[(x_A, x_B, 'A')]) > 0 and len(self.means_choice[(x_A, x_B, 'B')]) > 0:
                self.previous['split'][choice].append(means[key][step_range[0]:step_range[1]])
            else:
                if len(means) > 0:
                    self.previous['easy'][choice].append(means[key][step_range[0]:step_range[1]])
        return (self.mean_window(np.mean(self.previous['easy']['A'], axis=0)),
                self.mean_window(np.mean(self.previous['easy']['B'], axis=0)),
                self.mean_window(np.mean(self.previous['split']['A'], axis=0)),
                self.mean_window(np.mean(self.previous['split']['B'], axis=0)))

    def regression_hysteresis(self, type=None, ΔA=(0, 20), ΔB=(0, 20)):
        # hysteresis
        X_A, X_B, choice_B = {'easy': [], 'split': []}, {'easy': [], 'split': []}, {'easy': [], 'split': []}
        for (x_a, x_b), (n_a, n_b) in sorted(self.choices.items()):
            if x_a != 0 or x_b != 0:
                if n_a != 0 and n_b != 0:
                    X_A['split'].append(x_a)
                    X_B['split'].append(x_b)
                    choice_B['split'].append(n_b / (n_a + n_b))
                else:
                    X_A['easy'].append(x_a)
                    X_B['easy'].append(x_b)
                    choice_B['easy'].append(n_b / (n_a + n_b))
        a_opt_easy, a_cov_easy = scipy.optimize.curve_fit(self.approx_polynome, [X_A['easy'], X_B['easy']],
                                                          choice_B['easy'], bounds=((-20,) * 6, (20,) * 6))
        a_opt_split, a_cov_split = scipy.optimize.curve_fit(self.approx_polynome, [X_A['split'], X_B['split']],
                                                            choice_B['split'], bounds=((-20,) * 6, (20,) * 6))

        # computing the regressed model over all possible quantities by 0.5 increments.
        X_A_reg = np.arange(ΔA[0], ΔA[1] + 0.5, 0.5)
        X_B_reg = np.arange(ΔB[0], ΔB[1] + 0.5, 0.5)
        X_A_reg, X_B_reg = np.meshgrid(X_A_reg, X_B_reg)
        choice_B_reg_easy = 100 * self.approx_polynome((X_A_reg, X_B_reg), *a_opt_easy)
        choice_B_reg_split = 100 * self.approx_polynome((X_A_reg, X_B_reg), *a_opt_split)
        if type == 'easy':
            return X_A['easy'], X_B['easy'], 100 * np.array(choice_B['easy']), X_A_reg, X_B_reg, choice_B_reg_easy
        elif type == 'split':
            return X_A['split'], X_B['split'], 100 * np.array(choice_B['split']), X_A_reg, X_B_reg, choice_B_reg_split
        else:
            return ValueError

    def fig9_average_activity(self, offers):
        """
        For each trial, the average activity between 400ms and 600ms after the
        offer of CJA cells (y-axis) and CJB cells (x-axis).

        :param offers:  only trials that have an offer belonging to the
                        `offers` set will be considered.

        The data is returned as a dictionary offer -> list of x/y coordinates.
        """
        results = {offer: [] for offer in offers}
        step_range = self.step_range((0.4, 0.6))

        for offer, trials in self.model.history.trials.items():
            if offer in offers:
                for trial in trials:
                    r_cja_mean = np.mean(trial['r_1'][step_range[0]:step_range[1]])
                    r_cjb_mean = np.mean(trial['r_2'][step_range[0]:step_range[1]])
                    results[offer].append((r_cjb_mean, r_cja_mean))

        return results
