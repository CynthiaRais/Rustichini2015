# 1:CJA, 2:CJB, 3:NS


class TrialHistory:

    def __init__(self, model, x_a, x_b, full_log=False):
        self.x_a = x_a
        self.x_b = x_b
        self.full_log = full_log

        keys = ['r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb']
        if self.full_log:
            keys.extend(['S_ampa_1', 'S_ampa_2', 'S_ampa_3',
                         'S_nmda_1', 'S_nmda_2', 'S_nmda_3',
                         'S_gaba',
                         'I_eta_1', 'I_eta_2', 'I_eta_3', 'I_eta_I',
                         'I_ampa_rec_1', 'I_ampa_rec_2', 'I_ampa_rec_3', 'I_ampa_rec_I',
                         'phi_1', 'phi_2', 'phi_3', 'phi_I',
                         'I_syn_1', 'I_syn_2', 'I_syn_3', 'I_syn_I',
                         'I_stim_1', 'I_stim_2'
                        ])

        self.data = {name: [] for name in keys}
        for i in ['1', '2', '3', 'I']:
            self.data['r_{}'.format(i)].append(model.r[i])

        if self.full_log:
            for i in ['1', '2', '3', 'I']:
                self.data['I_eta_{}'.format(i)].append(0.0)
            for i in ['1', '2', '3']:
                self.data['S_ampa_{}'.format(i)].append(model.S_ampa[i])
                self.data['S_nmda_{}'.format(i)].append(model.S_nmda[i])

            self.data['S_gaba'].append(model.S_gaba)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, value):
        return self.data[value]

    def update(self, model, I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim, I_syn, phi):
        self.data['r_ova'].append(model.r_ov['1'])
        self.data['r_ovb'].append(model.r_ov['2'])

        for i in ['1', '2', '3', 'I']:
            self.data['r_{}'.format(i)].append(model.r[i])

        if self.full_log:
            for i in ['1', '2', '3', 'I']:
                self.data['I_eta_{}'.format(i)].append(model.I_eta[i])
                self.data['I_ampa_rec_{}'.format(i)].append(I_ampa_rec[i])
                self.data['I_syn_{}'.format(i)].append(I_syn[i])
                self.data['phi_{}'.format(i)].append(phi[i])

            for i in ['1', '2']:
                self.data['I_stim_{}'.format(i)].append(I_stim[i])

            for i in ['1', '2', '3']:
                self.data['S_ampa_{}'.format(i)].append(model.S_ampa[i])
                self.data['S_nmda_{}'.format(i)].append(model.S_nmda[i])

            self.data['S_gaba'].append(model.S_gaba)

    def export(self, keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
        """Export a trial history, filtering the desired keys"""
        return {key: self.data[key] for key in keys}


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
