# 1:CJA, 2:CJB, 3:NS


class TrialHistory:

    def __init__(self, model, x_a, x_b):
        self.x_a = x_a
        self.x_b = x_b

        self.data = {name: [] for name in ['r_1', 'r_2', 'r_3', 'r_I',
                                           'r_ova', 'r_ovb',
                                           'S_ampa_1', 'S_ampa_2', 'S_ampa_3',
                                           'S_nmda_1', 'S_nmda_2', 'S_nmda_3',
                                           'S_gaba',
                                           'I_ampa_rec_1', 'I_ampa_rec_2', 'I_ampa_rec_3', 'I_ampa_rec_I',
                                           'phi_1', 'phi_2', 'phi_3', 'phi_I',
                                           'Isyn_1', 'Isyn_2', 'Isyn_3', 'Isyn_I',
                                           'I_stim_1', 'I_stim_2'
                                          ]}
        for i in ['1', '2', '3', 'I']:
            getattr(self, 'r_{}'.format(i)).append(model.r[i])

        for i in ['1', '2', '3']:
            getattr(self, 'S_ampa_{}'.format(i)).append(model.S_ampa[i])
            getattr(self, 'S_nmda_{}'.format(i)).append(model.S_nmda[i])

        self.S_gaba.append(model.S_gaba)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, name):
        return self.data[name]

    def update(self, model, I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim, I_syn, phi):
        for i in ['1', '2', '3', 'I']:
            getattr(self, 'r_{}'.format(i)).append(model.r[i])
            getattr(self, 'I_ampa_rec_{}'.format(i)).append(I_ampa_rec[i])
            getattr(self, 'phi_{}'.format(i)).append(phi[i])
            getattr(self, 'Isyn_{}'.format(i)).append(I_syn[i])

        for i in ['1', '2']:
            getattr(self, 'I_stim_{}'.format(i)).append(I_stim[i])

        for i in ['1', '2', '3']:
            getattr(self, 'S_ampa_{}'.format(i)).append(model.S_ampa[i])
            getattr(self, 'S_nmda_{}'.format(i)).append(model.S_nmda[i])

        self.S_gaba[0] = model.S_gaba

        self.r_ova.append(model.r_ov['1'])
        self.r_ovb.append(model.r_ov['2'])

    def export(self, keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
        """Export a trial history, filtering the desired keys"""
        return {key: getattr(self, key) for key in keys}


class History:
    """Hold the history of multiple trials"""

    def __init__(self, model, keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
        self.model = model # keeping all the parameters
        self.keys = keys
        self.reset()

    def reset(self):
        self.trials = {(x_A, x_B): [] for x_A in range(self.model.ΔA + 1)
                                      for x_B in range(self.model.ΔB + 1)}
        self.trials_choice = {(x_A, x_B, choice): [] for x_A in range(self.model.ΔA + 1)
                                                     for x_B in range(self.model.ΔB + 1)
                                                     for choice in ['A', 'B']}

    def add_trial(self, trial_history):
        key = (trial_history.x_a, trial_history.x_b)
        self.trials[key].append(trial_history.export(keys=self.keys))
        key_choice = (trial_history.x_a, trial_history.x_b, trial_history.choice)
        self.trials_choice[key_choice].append(trial_history.export(keys=self.keys))

    def clear(self):
        self.trials, self.trials_choice = None, None
