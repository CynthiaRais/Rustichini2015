# 1:CJA, 2:CJB, 3:NS


class TrialHistory:

    def __init__(self, model):

        self.data = {name: [0] for name in ['r_1', 'r_2', 'r_3', 'r_I',
                                            'S_ampa_1', 'S_ampa_2', 'S_ampa_3',
                                            'S_nmda_1', 'S_nmda_2', 'S_nmda_3',
                                            'S_gaba',
                                            'I_ampa_rec_1', 'I_ampa_rec_2', 'I_ampa_rec_3', 'I_ampa_rec_I',
                                            'phi_1', 'phi_2', 'phi_3', 'phi_I',
                                            'Isyn_1', 'Isyn_2', 'Isyn_3', 'Isyn_I',
                                            'I_stim_1', 'I_stim_2'
                                            ]}
        for i in ['1', '2', '3', 'I']:
            getattr(self, 'r_{}'.format(i))[0] = model.r[i]

        for i in ['1', '2', '3']:
            getattr(self, 'S_ampa_{}'.format(i))[0] = model.S_ampa[i]
            getattr(self, 'S_nmda_{}'.format(i))[0] = model.S_nmda[i]

        self.S_gaba[0] = model.S_gaba


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
