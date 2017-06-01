#-*- coding: utf-8 -*-

import numpy as np


class Model:

    def __init__(self,  N_E         = 1600,
                        N_I         = 400,
                        C_ext       = 800,
                        f           = 0.15,
                        r_ext       = 3,               # spike/s

                        τ_ampa         = 0.002,        # s
                        τ_nmda         = 0.100,        # s
                        τ_gaba         = 0.005,        # s
                        J_ampa_ext_pyr = -0.1123,
                        J_ampa_rec_pyr = -0.0027,
                        J_nmda_rec_pyr = -0.00091979,
                        J_gaba_rec_pyr = 0.0215,
                        J_ampa_ext_in  = -0.0842,
                        J_ampa_rec_in  = -0.0022,
                        J_nmda_rec_in  = -0.00083446,
                        J_gaba_rec_in  = 0.0180,
                        ΔJ             = 30,
                        γ              = 0.641,
                        σ_eta          = 0.020,

                        I_E = 125,
                        g_E = 0.16,
                        c_E = 310,
                        I_I = 177,
                        g_I = 0.087,
                        c_I = 615,

                        r_o     = 0,                    # spike/s (0 or 6)
                        Δ_r     = 8,                    # spike/s
                        t_offer = 1.0,                  # s
                        t_exp = 2.0,                    # s
                        dt=0.0005,                      # s
                        n = 4000,                       # number of trials

                        δ_J_hl   = (1 ,1),
                        δ_J_stim = (2 ,1),
                        δ_J_gaba = (1 ,1 ,1),
                        δ_J_nmda = (1 ,1),

                        w_p = 1.75,
                        a = 0.175,
                        b = 0.030,
                        c = 0.400,
                        d = 0.100,

                        ΔA = 20,
                        ΔB = 20,

                        x_max_list = None,
                        x_min_list = None,

                        random_seed = 0,
                        verbose     = False):

        np.random.seed(random_seed) # FIXME: per model Random instance
        self.verbose = verbose

        # Network parameters
        self.N_E, self.N_I = N_E, N_I
        self.C_ext, self.f, self.r_ext = C_ext, f, r_ext

        # Time constants, synaptic efficacies, and noise
        self.τ_ampa, self.τ_nmda, self.τ_gaba = τ_ampa, τ_nmda, τ_gaba
        self.J_ampa_ext_pyr, self.J_ampa_rec_pyr = J_ampa_ext_pyr, J_ampa_rec_pyr
        self.J_nmda_rec_pyr, self. J_gaba_rec_pyr = J_nmda_rec_pyr, J_gaba_rec_pyr
        self.J_ampa_ext_in, self.J_ampa_rec_in = J_ampa_ext_in, J_ampa_rec_in
        self.J_nmda_rec_in, self.J_gaba_rec_in = J_nmda_rec_in, J_gaba_rec_in
        self.γ, self.σ_eta = γ, σ_eta

        # Parameters of input-output function for integrate-and-fire neurons
        self.I_E, self.g_E, self.c_E = I_E, g_E, c_E
        self.I_I, self.g_I, self.c_I = I_I, g_I, c_I

        # Parameters used to model OV cells
        self.r_o, self.Δ_r, self.t_offer = r_o, Δ_r, t_offer
        self.a = a + t_offer                        # s
        self.b = b                                  # s
        self.c = t_offer + c                        # s
        self.d = d                                  # s
        self.J_ampa_input = ΔJ * J_ampa_ext_pyr

        self.w_p = w_p
        self.w_m = 1 - f * (self.w_p - 1) / (1 - self.f)

        # Hebbian learning and synaptic imbalance
        self.δ_J_hl_cj_a, self.δ_J_hl_cj_b = δ_J_hl[0], δ_J_hl[1]
        self.δ_J_stim_cj_a, self.δ_J_stim_cj_b = δ_J_stim[0], δ_J_stim[1]
        self.δ_J_gaba_cj_a, self.δ_J_gaba_cj_b, self.δ_J_gaba_ns = δ_J_gaba[0], δ_J_gaba[1], δ_J_gaba[2]
        self.δ_J_nmda_cj_a, self.δ_J_nmda_cj_b = δ_J_nmda[0], δ_J_nmda[1]

        # Parameters of the experience
        self.t_exp = t_exp
        self.dt = dt
        self.n = n
        self.ΔA = ΔA
        self.ΔB = ΔB
        self.list_choice = ['A', 'B']

        # Values at the beginning of each trial (before figure 7)
        self.r_i_cj_a, self.r_i_cj_b, self.r_i_ns, self.r_i_cv_cells = 0, 0, 0, 0
        self.I_eta_cj_a, self.I_eta_cj_b, self.I_eta_ns, self.I_eta_cv = 0, 0, 0, 0
        self.S_cj_a, self.S_cj_b, self.S_ns = [0, 0, 0], [0, 0, 0], [0, 0, 0]
        self.S_gaba_cv = 0
        self.choice = 0

        #self.result = sorted(self.result.items(), key = operator.itemgetter(0))

        # self.gmax = np.max((1 / (1 + np.exp(- (t- self.a) / self.b))) * (1 / (1 + np.exp((t - self.c) / self.d)))
        #                   for t in np.arange(0, 2.0, self.dt))         #marche?

        # Determination of g maximum for OV cells firing rate
        self.list_g = []
        for t in np.arange(0, self.t_exp, dt):
            self.g = (1 / (1 + np.exp(- (t - self.a) / self.b))) * (1 / (1 + np.exp((t - self.c) / self.d)))
            self.list_g.append(self.g)
        self.g_max = np.max(self.list_g)

        self.x_min_list = x_min_list
        self.x_max_list = x_max_list
        self.ov, self.cjb, self.cv = {}, {}, {}

        # to keep firing rate of one trial
        self.result_one_trial = {}
        for i in range(self.ΔA + 1):
            for j in range(self.ΔB + 1):
                for choice_i in self.list_choice:
                    self.result_one_trial[(i, j, choice_i)] = []
        self.ov_b_one_trial, self.r_i_cj_a_one_trial, self.r_i_cj_b_one_trial = [], [], []
        self.r_i_ns_one_trial, self.r_i_cv_cells_one_trial = [], []

        # Control of noise
        self.I_eta_cj_a_list = []
        self.I_eta_cj_b_list = []
        self.I_eta_cj_ns_list = []
        self.I_eta_cj_cv_list = []

        if self.verbose:
            print("Finished model initialization.")


    def firing_rate_pyr_cells(self, r_i, phi): #1
        """Update the firing rate of pyramidale cells (eq. 1)"""
        r_i += ((- r_i + phi) / self.τ_ampa) * self.dt
        return r_i

    def firing_rate_I(self, r_I, phi):  # 2
        """Update the firing rate of interneurons (eq. 2)"""
        r_I += ((-r_I + phi) / self.τ_ampa) * self.dt
        return r_I


    def channel_ampa(self, S_ampa, r_i):  # 3
        """Open AMPA channels (eq. 3)"""
        S_ampa += ((- S_ampa / self.τ_ampa) + r_i) * self.dt
        return S_ampa

    def channel_nmda(self, S_nmda, r_i):  # 4
        """Open NMDA channels (eq. 4)"""
        S_nmda += ((-S_nmda / self.τ_nmda) + (1 - S_nmda) * self.γ * r_i) * self.dt
        return S_nmda

    def channel_gaba(self, S_gaba, r_I):  # 5
        """Open GABA channels (eq. 5)"""
        S_gaba += (-S_gaba / self.τ_gaba + r_I) * self.dt
        return S_gaba

    def Φ(self, I_syn, c, i, gain):  # 6
        """Input-ouput relation for leaky integrate-and-fire cell (Abbott and Chance, 2005) (eq. 6)"""
        phi = ((c * I_syn - i) / (1 - np.exp(-gain * (c * I_syn - i))))

        return phi


        ## Currents and parameters

    def I_syn(self, I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim):  # 7
        """Compute the input current for pyramidal cells (eq. 7)"""
        return I_ampa_ext + I_ampa_rec + I_nmda_rec + I_gaba_rec + I_stim

    def I_ampa_ext(self, I_eta):  # 8
        """Compute the external AMPA current for pyramidal cells (eq. 8)"""
        return -self.J_ampa_ext_pyr * self.τ_ampa * self.C_ext * self.r_ext + I_eta

    def I_ampa_rec(self, S_ampa_1, S_ampa_2, S_ampa_3):  # 9
        """Compute the recurrent AMPA current for CJA and CJB cells (eq. 9)"""
        return (-self.N_E * self.f * self.J_ampa_rec_pyr * (self.w_p * S_ampa_1 + self.w_m * S_ampa_2)
                - self.N_E * (1 - 2 * self.f) * self.J_ampa_rec_pyr * self.w_m * S_ampa_3)

    def I_ampa_rec_3(self, S_ampa_1, S_ampa_2, S_ampa_3):  # 10
        """Compute the recurrent AMPA current for NS cells (eq. 10)"""
        return (-self.N_E * self.f * self.J_ampa_rec_pyr * (S_ampa_1 + S_ampa_2)
                - self.N_E * (1 - 2 * self.f) * self.J_ampa_rec_pyr * S_ampa_3)

    def I_nmda_rec(self, δ_J_ndma, S_nmda_1, S_nmda_2, S_nmda_3):  # 11
        """Compute the recurrent NMDA current for CJA and CJB cells (eq. 11)"""
        return (-self.N_E * self.f * self.J_nmda_rec_pyr * δ_J_ndma * (self.w_p * S_nmda_1 + self.w_m * S_nmda_2)
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_pyr * self.w_m * S_nmda_3)

    def I_nmda_rec_3(self, S_nmda_1, S_nmda_2, S_nmda_3):  # 12
        """Compute the recurrent NMDA current for NS cells (eq. 12)"""
        return (-self.N_E * self.f * self.J_nmda_rec_pyr * (S_nmda_1 + S_nmda_2)
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_pyr * S_nmda_3)

    def I_gaba_rec(self, δ_J_gaba, S_gaba):  # 13
        """Compute the recurrent NMDA current for pyramidal cells (eq. 13)"""
        return -self.N_I * self.J_gaba_rec_pyr * δ_J_gaba * S_gaba


    def I_ampa_ext_I(self, I_eta):  # 14
        """Compute the external AMPA current for interneurons (eq. 14)"""
        return -self.J_ampa_ext_in * self.τ_ampa * self.C_ext * self.r_ext + I_eta

    def I_ampa_rec_I(self, S_ampa_1, S_ampa_2, S_ampa_3):  # 15
        """Compute the recurrent AMPA current for interneurons (eq. 15)"""
        return (-self.N_E * self.f * self.J_ampa_rec_in * (S_ampa_1 + S_ampa_2)
                - self.N_E * (1 - 2 * self.f) * self.J_ampa_rec_in * S_ampa_3)

    def I_nmda_rec_I(self, S_nmda_1, S_nmda_2, S_nmda_3):  # 16
        """Compute the recurrent NMDA current for interneurons (eq. 16)"""
        return (-self.N_E * self.f * self.J_nmda_rec_in * (S_nmda_1 + S_nmda_2)
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_in * S_nmda_3)

    def I_gaba_rec_I(self, S_gaba):  # 17
        """Compute the recurrent GABA current for interneurons (eq. 17)"""
        return -self.N_I * self.J_gaba_rec_in * S_gaba

    def eta(self):
        """Ornstein-Uhlenbeck process (here just Gaussian random noise)"""
        return np.random.normal(0, 1)

    def white_noise(self, I_eta):  # 18
        """Update I_eta, the noise term (eq. 18)"""
        I_eta += -I_eta * (self.dt / self.τ_ampa) + self.eta() * np.sqrt(self.dt/self.τ_ampa) * self.σ_eta
        return I_eta

    def I_stim(self, δ_j_hl, δ_j_stim, r_ov):  # 19
        """Computing the primary input (eq. 19)"""
        return -self.J_ampa_input * δ_j_hl * δ_j_stim * self.τ_ampa * r_ov


    def firing_ov_cells(self, x, xmin, x_max, t):  # 20, 21, 22, 23
        """Computing the activity profile of OV cells (eq. 20, 21, 22, 23)"""
        x_i = (x - xmin) / (x_max - xmin)
        g_t = (1 / (1 + np.exp(- (t - self.a) / self.b))) * (1 / (1 + np.exp((t - self.c) / self.d)))

        f_t = g_t / self.g_max
        #assert f_t <= 1, 'firing ov trop haut'
        r_ov = self.r_o + self.Δ_r * f_t * x_i
        return r_ov


    def logistic_model(self, a_0, a_1, a_2, a_3, a_4, a_5, quantity_a, quantity_b):  # 25
        """Computing the logistic model of figure 4 to examine departures from linearity"""
        X = a_0 + a_1 * quantity_a + a_2 * quantity_b + a_3 * (quantity_a) ** 2 + a_4 * (quantity_b) ** 2 + a_5 * (
        quantity_a * quantity_b)
        choice_B = 1 / (1 + np.exp(-X))
        return choice_B

    ##

    def cj_cells(self, r_i_cj, S_ampa_cj, S_nmda_cj, S_gaba_cj, I_eta_cj,
                 S_ampa_cj_2, S_ampa_ns, S_nmda_cj_2, S_nmda_ns, r_i_cv_cells, r_ov,
                 δ_J_ndma, δ_J_gaba, δ_J_hl, δ_J_stim):
        """Compute firing rate of CJA and CJB cells"""

        S_ampa_cj = self.channel_ampa(S_ampa_cj, r_i_cj)  # equation 3
        S_nmda_cj = self.channel_nmda(S_nmda_cj, r_i_cj)  # equation 4
        S_gaba_cj = self.channel_gaba(S_gaba_cj, r_i_cv_cells)  # equation 5
        S_cj = [S_ampa_cj, S_nmda_cj, S_gaba_cj]

        I_eta_cj = self.white_noise(I_eta_cj)  # equation 18
        I_ampa_ext_cj = self.I_ampa_ext(I_eta_cj)  # equation 8
        I_ampa_rec_cj = self.I_ampa_rec(S_ampa_cj, S_ampa_cj_2, S_ampa_ns)  # equation 9
        I_nmda_rec_cj = self.I_nmda_rec(δ_J_ndma, S_nmda_cj, S_nmda_cj_2, S_nmda_ns)  # equation 11
        I_gaba_rec_cj = self.I_gaba_rec(δ_J_gaba, S_gaba_cj)  # equation 13
        I_stim_cj = self.I_stim(δ_J_hl, δ_J_stim, r_ov)  # equation 19
        I_syn_cj = self.I_syn(I_ampa_ext_cj, I_ampa_rec_cj, I_nmda_rec_cj, I_gaba_rec_cj, I_stim_cj)  # equation 7

        phi_cj = self.Φ(I_syn_cj, self.c_E, self.I_E, self.g_E)  # equation 6
        r_i_cj = self.firing_rate_pyr_cells(r_i_cj, phi_cj)  # equation 1

        #assert r_i_cj <=100
        return r_i_cj, S_cj

    def ns_cells(self, r_i_ns, S_ampa_ns, S_nmda_ns, S_gaba_ns, I_eta_ns,
                 S_ampa_cj_a, S_ampa_cj_b, S_nmda_cj_a, S_nmda_cj_b, r_i_cv_cells,
                 δ_J_gaba):
        """Compute firing rate of NS cells"""

        S_ampa_ns = self.channel_ampa(S_ampa_ns, r_i_ns)  # equation 3
        S_nmda_ns = self.channel_nmda(S_nmda_ns, r_i_ns)  # equation 4
        S_gaba_ns = self.channel_gaba(S_gaba_ns, r_i_cv_cells)  # equation 5
        S_ns = [S_ampa_ns, S_nmda_ns, S_gaba_ns]

        I_eta_ns = self.white_noise(I_eta_ns)  # equation 18
        I_ampa_ext_ns = self.I_ampa_ext(I_eta_ns)  # equation 8
        I_ampa_rec_ns = self.I_ampa_rec_3(S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns)  # equation 10
        I_nmda_rec_ns = self.I_nmda_rec_3(S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns)  # equation 12
        I_gaba_rec_ns = self.I_gaba_rec(δ_J_gaba, S_gaba_ns)  # equation 13
        I_stim_ns = 0
        I_syn_ns = self.I_syn(I_ampa_ext_ns, I_ampa_rec_ns, I_nmda_rec_ns, I_gaba_rec_ns, I_stim_ns)  # equation 7

        phi_ns = self.Φ(I_syn_ns, self.c_E, self.I_E, self.g_E)  # equation 6
        r_i_ns = self.firing_rate_pyr_cells(r_i_ns, phi_ns)  # equation 1

        return r_i_ns, S_ns

    def cv_cells(self,r_i_cv_cells, S_gaba_cv, I_eta_cv,
                 S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns,
                 S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns):
        """Compute firing rate of CV cells"""

        S_gaba_cv = self.channel_gaba(S_gaba_cv, r_i_cv_cells)  # equation 5

        I_eta_cv = self.white_noise(self.I_eta_cv)  # equation 18
        I_ampa_ext_cv = self.I_ampa_ext_I(I_eta_cv)  # equation 14
        I_ampa_rec_cv = self.I_ampa_rec_I(S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns)  # equation 15
        I_nmda_rec_cv = self.I_nmda_rec_I(S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns)  # equation 16
        I_gaba_rec_cv = self.I_gaba_rec_I(S_gaba_cv)  # equation 17
        I_stim_cv = 0
        I_syn_cv_cells = self.I_syn(I_ampa_ext_cv, I_ampa_rec_cv, I_nmda_rec_cv, I_gaba_rec_cv, I_stim_cv)  # equation 7

        phi_cv_cells = self.Φ(I_syn_cv_cells, self.c_I, self.I_I, self.g_I)  # equation 6
        r_i_cv_cells = self.firing_rate_I(r_i_cv_cells, phi_cv_cells)  # equation 2
        #assert r_i_cv_cells <= 80, 'r_i_cv = {}'.format(r_i_cv_cells)
        return r_i_cv_cells, S_gaba_cv


    def one_trial(self, x_a, x_b):
        """Compute one trial"""

        # Firing rate of OV B cell, CJ B cell and CV cell for one trial
        self.r_i_cj_a, self.r_i_cj_b, self.r_i_ns, self.r_i_cv_cells = 3, 3, 3, 8
        self.I_eta_cj_a, self.I_eta_cj_b, self.I_eta_ns, self.I_eta_cv = 0, 0, 0, 0
        self.S_cj_a, self.S_cj_b, self.S_ns = [0, 0.1, 0], [0, 0.1, 0], [0, 0.1, 0]
        self.S_gaba_cv = 0
        self.choice = 0
        self.ov_b_one_trial, self.r_i_cj_a_one_trial, self.r_i_cj_b_one_trial = [], [], []
        self.r_i_ns_one_trial, self.r_i_cv_cells_one_trial = [], []

        for t in np.arange(0, self.t_exp + self.dt, self.dt):

            """Firing rate of OV cells"""
            r_ov_a = self.firing_ov_cells(x_a, self.x_min_list[0], self.x_max_list[0], t)
            r_ov_b = self.firing_ov_cells(x_b, self.x_min_list[1], self.x_max_list[1], t)
            #assert r_ov_a <= 8, 'r_ov_a = {}'.format(r_ov_a)
            #assert r_ov_b <= 8, 'r_ov_b = {}'.format(r_ov_b)

            """Firing rate of CJA and CJB cells"""
            self.r_i_cj_a, self.S_cj_a = self.cj_cells(self.r_i_cj_a, self.S_cj_a[0], self.S_cj_a[1], self.S_cj_a[2],
                                                       self.I_eta_cj_a, self.S_cj_b[0], self.S_ns[0], self.S_cj_b[1],
                                                       self.S_ns[1], self.r_i_cv_cells, r_ov_a,self.δ_J_nmda_cj_a,
                                                       self.δ_J_gaba_cj_a, self.δ_J_hl_cj_a, self.δ_J_stim_cj_a)
            #assert r_i_cj_a <= 80, 'r_i_cj = {0}, t ={1}'.format(r_i_cj_a, t)

            self.r_i_cj_b, self.S_cj_b = self.cj_cells(self.r_i_cj_b, self.S_cj_b[0], self.S_cj_b[1], self.S_cj_b[2],
                                                       self.I_eta_cj_b, self.S_cj_a[0], self.S_ns[0], self.S_cj_a[1],
                                                       self.S_ns[1], self.r_i_cv_cells, r_ov_b,self.δ_J_nmda_cj_b,
                                                       self.δ_J_gaba_cj_b, self.δ_J_hl_cj_b, self.δ_J_stim_cj_b)
            #assert r_i_cj_b <= 80, 'r_i_cj = {}, t ={1}'.format(r_i_cj_b, t)

            """Firing rate of NS cells"""
            self.r_i_ns, self.S_ns = self.ns_cells(self.r_i_ns, self.S_ns[0], self.S_ns[1], self.S_ns[2],
                                                   self.I_eta_ns, self.S_cj_a[0], self.S_cj_b[0], self.S_cj_a[1],
                                                   self.S_cj_b[1], self.r_i_cv_cells,self.δ_J_gaba_ns)

            """Firing rate of CV cells"""
            self.r_i_cv_cells, self.S_gaba_cv = self.cv_cells(self.r_i_cv_cells, self.S_gaba_cv,
                                                              self.I_eta_cv, self.S_cj_a[0], self.S_cj_b[0],
                                                              self.S_ns[0], self.S_cj_a[1], self.S_cj_b[1], self.S_ns[1])

            self.ov_b_one_trial.append(r_ov_b)
            self.r_i_cj_a_one_trial.append(self.r_i_cj_a)
            self.r_i_cj_b_one_trial.append(self.r_i_cj_b)
            self.r_i_ns_one_trial.append(self.r_i_ns)
            self.r_i_cv_cells_one_trial.append(self.r_i_cv_cells)
            #le bruit
            self.I_eta_cj_a_list.append(self.I_eta_cj_a)
            self.I_eta_cj_b_list.append(self.I_eta_cj_b)
            self.I_eta_cj_ns_list.append(self.I_eta_ns)
            self.I_eta_cj_cv_list.append(self.I_eta_cv)


        # Determine the final choice in the time window 400-600ms after the offer
        ria, rib = 0, 0
        for i in range(2800, 3201):
            ria += self.r_i_cj_a_one_trial[i]
            rib += self.r_i_cj_b_one_trial[i]
            if ria < rib:
                self.choice = 'B'
            else:
                self.choice = 'A'

        if self.choice != 'A' and self.choice != 'B':
            raise ValueError('no choice')

        return [self.ov_b_one_trial, self.r_i_cj_a_one_trial,
                self.r_i_cj_b_one_trial, self.r_i_ns_one_trial, self.r_i_cv_cells_one_trial, self.choice,
                self.I_eta_cj_a_list, self.I_eta_cj_b_list, self.I_eta_cj_ns_list, self.I_eta_cj_cv_list]

    def save_history(self, data):
        print("Saving history...")
        np.save(data, self.result_one_trial)
        print("done.")


if __name__ == "__main__":
    Class = Model()
    Class.one_trial()
