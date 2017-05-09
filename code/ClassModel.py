#-*- coding: utf-8 -*-

import numpy as np
import math
import operator # to sort results
import bokeh
import bokeh.plotting


class Economic_Decisions_Model:

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
                        n = 4000,                        # number of trials

                        δ_J_hl = (1 ,1),
                        δ_J_stim = (2 ,1),
                        δ_J_gaba = (1 ,1 ,1),
                        δ_J_nmda = (1 ,1),

                        w_p=1.75,
                        a = 0.175,
                        b = 0.030,
                        c = 0.400,
                        d = 0.100):


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
        self.J_ampa_input = 30 * J_ampa_ext_pyr

        self.w_p = w_p
        self.w_m = 1 - f * (self.w_p - 1) / (1 - self.f)

        # Hebbian learning and synaptic imbalance
        self.δ_J_hl_cj_a, self.δ_J_hl_cj_b = δ_J_hl[0], δ_J_hl[1]
        self.δ_J_stim_cj_a, self.δ_J_stim_cj_b = δ_J_stim[0], δ_J_stim[1]
        self.δ_J_gaba_cj_a, self.δ_J_gaba_cj_b, self.δ_J_gaba_ns = δ_J_gaba[0], δ_J_gaba[1], δ_J_gaba[2]
        self.δ_J_nmda_cj_a, self.δ_J_nmda_cj_b = δ_J_nmda[0], δ_J_nmda[1]

        self.t_exp = t_exp
        self.dt = dt
        self.n = n
        self.list_choice = ['A', 'B']


        #self.result = sorted(self.result.items(), key = operator.itemgetter(0))

        # self.gmax = np.max((1 / (1 + np.exp(- (t- self.a) / self.b))) * (1 / (1 + np.exp((t - self.c) / self.d)))
        #                   for t in np.arange(0, 2.0, self.dt))         #marche?

        # Determination of g maximum for OV cells firing rate
        self.list_g = []
        for t in np.arange(0, self.t_exp, dt):
            self.g = (1 / (1 + np.exp(- (t - self.a) / self.b))) * (1 / (1 + np.exp((t - self.c) / self.d)))
            self.list_g.append(self.g)
        self.g_max = np.max(self.list_g)

        self.quantity_a, self.quantity_b = [], [] # list of juice quantity A and B
        self.x_min_list, self.x_max_list = [], [] # list of minimum and maximum of juice A and B in a session
        self.ov, self.cjb, self.cv = {}, {}, {}



    def firing_rate_pyr_cells(self, r_i, phi): #1
        """Update the firing rate of pyramidale cells (eq. 1)"""
        r_i += ((- r_i + phi) / self.τ_ampa) * self.dt

        return r_i

    def firing_rate_I(self, r_I, phi):  # 2
        """Update the firing rate of interneurons (eq. 2)"""
        r_I += ((-r_I + phi) / self.τ_gaba) * self.dt
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
        I_eta += ((-I_eta + self.eta() * math.sqrt(self.τ_ampa * (self.σ_eta ** 2))) / self.τ_ampa) * self.dt
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

        assert r_i_cj <=100
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

        assert r_i_ns <=100, 'r_i_ns = {}'.format(r_i_ns)
        return r_i_ns, S_ns

    def cv_cells(self,r_i_cv_cells, S_gaba_cv, I_eta_cv,
                 S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns,
                 S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns):
        """Compute firing rate of CV cells"""

        S_gaba_cv = self.channel_gaba(S_gaba_cv, r_i_cv_cells)  # equation 5

        I_eta_cv = self.white_noise(I_eta_cv)  # equation 18
        I_ampa_ext_cv = self.I_ampa_ext_I(I_eta_cv)  # equation 14
        I_ampa_rec_cv = self.I_ampa_rec_I(S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns)  # equation 15
        I_nmda_rec_cv = self.I_nmda_rec_I(S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns)  # equation 16
        I_gaba_rec_cv = self.I_gaba_rec_I(S_gaba_cv)  # equation 17
        I_stim_cv = 0
        I_syn_cv_cells = self.I_syn(I_ampa_ext_cv, I_ampa_rec_cv, I_nmda_rec_cv, I_gaba_rec_cv, I_stim_cv)  # equation 7

        phi_cv_cells = self.Φ(I_syn_cv_cells, self.c_I, self.I_I, self.g_I)  # equation 6
        r_i_cv_cells = self.firing_rate_I(r_i_cv_cells, phi_cv_cells)  # equation 2
        assert r_i_cv_cells <= 80, 'r_i_cv = {}'.format(r_i_cv_cells)
        return r_i_cv_cells, S_gaba_cv


    ##

    def quantity_juice(self):
        # random choice of juice quantity, ΔA = ΔB = [0, 20]
        for i in range(self.n):
            self.x_a = np.random.randint(0, 21)
            self.x_b = np.random.randint(0, 21)
            while self.x_a == 0 and self.x_b == 0:
                p = np.random.random()
                if p < 0.5:
                    self.x_a = np.random.randint(0, 21)
                else:
                    self.x_b = np.random.randint(0, 21)
            self.quantity_a.append(self.x_a)
            self.quantity_b.append(self.x_b)
        self.x_min_list = [np.min(self.quantity_a)] + [np.min(self.quantity_b)]
        self.x_max_list = [np.max(self.quantity_a)] + [np.max(self.quantity_b)]
        return self.quantity_a, self.quantity_b, self.x_min_list, self.x_max_list


    def one_trial(self, x_a, x_b,
                  r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
                  I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
                  S_cj_a, S_cj_b, S_ns, S_gaba_cv):
        """Compute one trial"""

        # Firing rate of OV B cell, CJ B cell and CV cell for one trial
        ov_b_one_trial, r_i_cj_a_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial = [], [], [], []
        choice = 0
        for t in np.arange(0, self.t_exp + self.dt, self.dt):

            """Firing rate of OV cells"""
            r_ov_a = self.firing_ov_cells(x_a, self.x_min_list[0], self.x_max_list[0], t)
            r_ov_b = self.firing_ov_cells(x_b, self.x_min_list[1], self.x_max_list[1], t)
            assert r_ov_a <= 8, 'r_ov_a = {}'.format(r_ov_a)
            assert r_ov_b <= 8, 'r_ov_b = {}'.format(r_ov_b)

            """Firing rate of CJA and CJB cells"""
            r_i_cj_a, S_cj_a = self.cj_cells(r_i_cj_a, S_cj_a[0], S_cj_a[1], S_cj_a[2],
                                        I_eta_cj_a, S_cj_b[0], S_ns[0], S_cj_b[1], S_ns[1], r_i_cv_cells, r_ov_a,
                                             self.δ_J_nmda_cj_a, self.δ_J_gaba_cj_a, self.δ_J_hl_cj_a, self.δ_J_stim_cj_a)
            #assert r_i_cj_a <= 80, 'r_i_cj = {0}, t ={1}'.format(r_i_cj_a, t)

            r_i_cj_b, S_cj_b = self.cj_cells(r_i_cj_b, S_cj_b[0], S_cj_b[1], S_cj_b[2],
                                        I_eta_cj_b, S_cj_a[0], S_ns[0], S_cj_a[1], S_ns[1], r_i_cv_cells, r_ov_b,
                                             self.δ_J_nmda_cj_b, self.δ_J_gaba_cj_b, self.δ_J_hl_cj_b, self.δ_J_stim_cj_b)

            #assert r_i_cj_b <= 80, 'r_i_cj = {}, t ={1}'.format(r_i_cj_b, t)
            """Firing rate of NS cells"""
            r_i_ns, S_ns = self.ns_cells(r_i_ns, S_ns[0], S_ns[1], S_ns[2],
                                         I_eta_ns, S_cj_a[0], S_cj_b[0], S_cj_a[1], S_cj_b[1], r_i_cv_cells,
                                         self.δ_J_gaba_ns)

            """Firing rate of CV cells"""
            r_i_cv_cells, S_gaba_cv = self.cv_cells(r_i_cv_cells, S_gaba_cv,
                                                    I_eta_cv, S_cj_a[0], S_cj_b[0], S_ns[0], S_cj_a[1], S_cj_b[1], S_ns[1])

            ov_b_one_trial.append(r_ov_b)
            r_i_cj_a_one_trial.append(r_i_cj_a)
            r_i_cj_b_one_trial.append(r_i_cj_b)
            r_i_cv_cells_one_trial.append(r_i_cv_cells)

        """Determine the final choice"""
        ria, rib =0,0
        for i in range(2800, 3201):
            ria += r_i_cj_a_one_trial[i]
            rib += r_i_cj_b_one_trial[i]
            if (ria/400) < (rib/400):
                choice = 'B'
            else:
                choice = 'A'

        if choice != 'A' and choice != 'B':
            raise ValueError('no choice')
        print("choix final", x_a, x_b, choice, np.max(r_i_cj_a_one_trial), np.max(r_i_cj_b_one_trial), np.max(r_i_cv_cells_one_trial))
        return choice, ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial


    def session(self):
        self.quantity_a, self.quantity_b, self.x_min_list, self.x_max_list = self.quantity_juice()
        # Create and order the dictionary result
        result = {}  # to save mean of firing rates for each (quantity A, quantity B, choice)
        choice_B = {}
        for j in range(0, 21):
            for k in range(0, 21):
                for l in range(0, 2):
                    result[(j, k, self.list_choice[l])] = []
                    choice_B[(j,k)] = 0
        #print(len(result), len(result[(1, 1, 'A')]), len(result[(1, 14, 'B')]), "beginning")
        for i in range(self.n):
            '''for graphs until figure 7, all parameters are reset to 0 at the beginning of each trial'''
            """ Reset to zero of parameters before each trial"""
            r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells = 0, 0, 0, 0
            I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv = 0, 0, 0, 0
            S_cj_a, S_cj_b, S_ns = [0, 0, 0], [0, 0, 0], [0, 0, 0]
            S_gaba_cv = 0
            choice, ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial = self.one_trial(self.quantity_a[i], self.quantity_b[i],
                                                                                                r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
                                                                                                I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
                                                                                                S_cj_a, S_cj_b, S_ns, S_gaba_cv)
            if choice == 'B':
                choice_B[(self.quantity_a[i], self.quantity_b[i])] += 1
            """keep results in a dictionary"""
            if result[(self.quantity_a[i], self.quantity_b[i], choice)] == []:
                result[(self.quantity_a[i], self.quantity_b[i], choice)].append([ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial])

            else :
                for j in range(len(result[(self.quantity_a[i], self.quantity_b[i], choice)][0][0])):
                    result[(self.quantity_a[i], self.quantity_b[i], choice)][0][0][j] = (result[(self.quantity_a[i], self.quantity_b[i], choice)][0][0][j] + ov_b_one_trial[j]) / 2
                    result[(self.quantity_a[i], self.quantity_b[i], choice)][0][1][j] = (result[(self.quantity_a[i], self.quantity_b[i], choice)][0][1][j] + r_i_cj_b_one_trial[j]) / 2
                    result[(self.quantity_a[i], self.quantity_b[i], choice)][0][2][j] = (result[(self.quantity_a[i], self.quantity_b[i], choice)][0][2][j] + r_i_cv_cells_one_trial[j]) / 2

        return result, choice_B

    def result_firing_rate(self):

        """ on obtient la moyenne des ov_b rate en fonction du temps
        et si l'essai a eu une offre forte, moyenne, faible """
        result, choice_B = self.session()
        ovb_rate_low, ovb_rate_high, ovb_rate_medium = [], [], []
        mean_A_chosen_cj, mean_B_chosen_cj = [], []
        mean_low_cv, mean_medium_cv, mean_high_cv = [], [], []

        mean_ov_ij, mean_cjb_ij, mean_cv_ij = 0, 0, 0
        mean_ov_ji, mean_cjb_ji, mean_cv_ji = 0, 0, 0

        mean_ov_0B1A, mean_cj_0B1A, mean_cv_0B1A = 0, 0, 0
        mean_ov_1B0A, mean_cj_1B0A, mean_cv_1B0A = 0, 0, 0

        ''' le terme k représente le temps,
         le terme i représente la quantité de A,
          le j représente la quantité de B
          et le l représente la liste de l'essai l
          pour un temps donné, on ajoute les r_ov_b pour chaque (i,j) pour chaque essai l
          (figure 4A)'''
        
        for k in range(4001):
            mean_ov_low, mean_ov_high, mean_ov_medium = 0, 0, 0
            low, medium, high = 0, 0, 0
            for i in range(0, 21):
                for j in range(0, 7):
                    for choice_i in self.list_choice :
                        if result[(i, j, choice_i)] == []:
                            mean_ov_low += 0
                        else :
                            mean_ov_low += result[(i, j, choice_i)][0][0][k]
                            low += 1
                for j in range(7, 14):
                    for choice_i in self.list_choice:
                        if result[(i, j , choice_i)] == []:
                            mean_ov_medium +=0
                        else :
                            mean_ov_medium += result[(i, j, choice_i)][0][0][k]
                            medium += 1
                for j in range(14, 21):
                    for choice_i in self.list_choice:
                        if result[(i,j, choice_i)] == []:
                            mean_ov_high += 0
                        else :
                            mean_ov_high += result[(i, j, choice_i)][0][0][k]
                            high += 1
            ovb_rate_low.append(mean_ov_low / low)
            ovb_rate_medium.append(mean_ov_medium / medium)
            ovb_rate_high.append(mean_ov_high / high)

        '''mean depending on choice (figure 4E, 4I)'''
        for k in range(4002):
            A_chosen_cj, B_chosen_cj = 0, 0
            chosen_value_low, chosen_value_medium, chosen_value_high = 0, 0, 0
            A_nb, B_nb = 0, 0
            low_cv, medium_cv, high_cv = 0, 0, 0
            for i in range(21):
                for j in range(21):
                    for choice_i in self.list_choice:
                        if not len(result[(i, j, choice_i)]):
                            A_chosen_cj += 0
                            B_chosen_cj += 0
                            chosen_value_low += 0
                            chosen_value_medium += 0
                            chosen_value_high += 0
                        else :
                            if choice_i == 'A':
                                A_chosen_cj += result[(i, j, choice_i)][0][1][k]
                                A_nb +=1
                                if i < 7:
                                    chosen_value_low += result[(i, j, choice_i)][0][2][k]
                                    low_cv += 1
                                elif 7 < i < 14:
                                    chosen_value_medium += result[(i, j, choice_i)][0][2][k]
                                    medium_cv += 1
                                else:
                                    chosen_value_high += result[(i, j, choice_i)][0][2][k]
                                    high_cv += 1
                            else :
                                B_chosen_cj += result[(i, j, choice_i)][0][1][k]
                                B_nb +=1
                                if j < 7:
                                    chosen_value_low += result[(i, j, choice_i)][0][2][k]
                                    low_cv += 1
                                elif 7 < j < 14:
                                    chosen_value_medium += result[(i, j, choice_i)][0][2][k]
                                    medium_cv += 1
                                else:
                                    chosen_value_high += result[(i, j, choice_i)][0][2][k]
                                    high_cv += 1
            if A_nb == 0 : A_nb =1
            if B_nb == 0 : B_nb=1
            mean_A_chosen_cj.append(A_chosen_cj / A_nb)
            mean_B_chosen_cj.append(B_chosen_cj / B_nb)
            mean_low_cv.append(chosen_value_low / low_cv)
            mean_medium_cv.append(chosen_value_medium / medium_cv)
            mean_high_cv.append(chosen_value_high / high_cv)

        """ordonner le dico avant utilisation"""
        """4C, 4G, 4K"""
        for j in range(20, 3, -4):
            for choice_i in self.list_choice:
                self.ov[(1,j, choice_i)], self.cjb[(1, j, choice_i)], self.cv[(1, j, choice_i)] = [], [], []
                self.ov[(j,1, choice_i)], self.cjb[(j, 1, choice_i)], self.cv[(j, 1, choice_i)] = [], [], []
                self.ov[(1,0, choice_i)], self.cjb[(1,0, choice_i)], self.cv[(1,0, choice_i)] = [], [], []
                self.ov[(0,1, choice_i)], self.cjb[(0,1, choice_i)], self.cv[(0,1, choice_i)] = [], [], []

     #   for choice_i in self.list_choice:
     #       for i in range(1, 2):
     #           for j in range(20, 3, -4):
     #               if not len(result[(1, j, choice_i)]):
     #                   pass
     #               else :
     #                   for k in range (2000, 3001):
     #                       mean_ov_ij += result[(1, j, choice_i)][0][0][k]
     #                       mean_cjb_ij += result[(1, j, choice_i)][0][1][k]
     #                       mean_cv_ij += result[(1, j, choice_i)][0][2][k + 1000]
     #               if not len(result[(j, 1, choice_i)]) :
     #                   pass
     #               else :
     #                   for k in range(2000, 3001):
     #                       mean_ov_ji += result[(j, 1, choice_i)][0][0][k]
     #                       mean_cjb_ji += result[(j, 1, choice_i)][0][1][k]
     #                       mean_cv_ji += result[(j, 1, choice_i)][0][2][k + 1000]

     #               self.ov[(1,j,choice_i)].append(mean_ov_ij / 1000)
     #               self.ov[(j,i, choice_i)].append(mean_ov_ji / 1000)
     #               self.cjb[(1,j,choice_i)].append(mean_cjb_ij / 1000)
     #               self.cjb[(j,1, choice_i)].append(mean_cjb_ji / 1000)
     #               self.cv[(1,j, choice_i)].append(mean_cv_ij / 1000)
    #                self.cv[(j,1, choice_i)].append(mean_cv_ji / 1000)

     #       if not len(result[(1,0, choice_i)]):
     #           mean_ov_0B1A = 0
     #           mean_cj_0B1A = 0
     #           mean_cv_0B1A = 0
     #       else :
     #           for k in range(2000, 3001):
     #               mean_ov_0B1A += result[(1,0, choice_i)][0][0][k]
     #               mean_cj_0B1A += result[(1,0, choice_i)][0][1][k]
    #                mean_cv_0B1A += result[(1,0, choice_i)][0][2][k + 1000]

     #       self.ov[(0, 1, choice_i)].append(mean_ov_1B0A / 1000)
     #       self.cjb[(0, 1, choice_i)].append(mean_cj_1B0A / 1000)
     #       self.cv[(0, 1, choice_i)].append(mean_cv_1B0A / 1000)

     #       if not len(result[(0,1, choice_i)]):
     #           mean_ov_1B0A = 0
     #           mean_cj_1B0A = 0
     #           mean_cv_1B0A = 0
     #       else:
     #           for k in range(2000, 3001):
     #               mean_ov_1B0A += result[(0, 1, choice_i)][0][0][k]
     #               mean_cj_1B0A += result[(0, 1, choice_i)][0][1][k]
    #                mean_cv_1B0A += result[(0, 1, choice_i)][0][2][k + 1000]

            #self.ov[(1, 0, choice_i)].append(mean_ov_0B1A / 1000)
            #self.cjb[(1, 0, choice_i)].append(mean_cj_0B1A / 1000)
            #self.cv[(1, 0, choice_i)].append(mean_cv_0B1A / 1000)
        #########changement of method
        ov_A_choiceA, ov_A_choiceB, ov_B_choiceA, ov_B_choiceB = [], [], [], []
        cjb_A_choiceA, cjb_A_choiceB, cjb_B_choiceA, cjb_B_choiceB = [], [], [], []
        cv_A_choiceA, cv_A_choiceB, cv_B_choiceA, cv_B_choiceB = [], [], [], []
        ov_choiceA, cjb_choiceA, cv_choiceA = [], [], []
        ov_choiceB, cjb_choiceB, cv_choiceB = [], [], []
        for choice_i in self.list_choice:
            for i in range(1, 2):
                for j in range(20, 3, -4):
                    if not len(result[(1, j, choice_i)]):
                        pass
                    else:
                        for k in range(2000, 3001):
                            mean_ov_ij += result[(1, j, choice_i)][0][0][k]
                            mean_cjb_ij += result[(1, j, choice_i)][0][1][k]
                            mean_cv_ij += result[(1, j, choice_i)][0][2][k + 1000]
                    if not len(result[(j, 1, choice_i)]):
                        pass
                    else:
                        for k in range(2000, 3001):
                            mean_ov_ji += result[(j, 1, choice_i)][0][0][k]
                            mean_cjb_ji += result[(j, 1, choice_i)][0][1][k]
                            mean_cv_ji += result[(j, 1, choice_i)][0][2][k + 1000]
                    if choice_i == 'A':
                        ov_A_choiceA.append(mean_ov_ij / 1000)
                        ov_B_choiceA.append(mean_ov_ji / 1000)
                        cjb_A_choiceA.append(mean_cjb_ij / 1000)
                        cjb_B_choiceA.append(mean_cjb_ji / 1000)
                        cv_A_choiceA.append(mean_cv_ij / 1000)
                        cv_B_choiceA.append(mean_cv_ji / 1000)
                    elif choice_i == 'B':
                        ov_A_choiceB.append(mean_ov_ij / 1000)
                        ov_B_choiceB.append(mean_ov_ji / 1000)
                        cjb_A_choiceB.append(mean_cjb_ij / 1000)
                        cjb_B_choiceB.append(mean_cjb_ji / 1000)
                        cv_A_choiceB.append(mean_cv_ij / 1000)
                        cv_B_choiceB.append(mean_cv_ji / 1000)

            if not len(result[(1, 0, choice_i)]):
                mean_ov_0B1A = 0
                mean_cj_0B1A = 0
                mean_cv_0B1A = 0
            else:
                for k in range(2000, 3001):
                    mean_ov_0B1A += result[(1, 0, choice_i)][0][0][k]
                    mean_cj_0B1A += result[(1, 0, choice_i)][0][1][k]
                    mean_cv_0B1A += result[(1, 0, choice_i)][0][2][k + 1000]

            if not len(result[(0, 1, choice_i)]):
                mean_ov_1B0A = 0
                mean_cj_1B0A = 0
                mean_cv_1B0A = 0
            else:
                for k in range(2000, 3001):
                    mean_ov_1B0A += result[(0, 1, choice_i)][0][0][k]
                    mean_cj_1B0A += result[(0, 1, choice_i)][0][1][k]
                    mean_cv_1B0A += result[(0, 1, choice_i)][0][2][k + 1000]
            if choice_i == 'A' :
                ov_choiceA = [mean_ov_0B1A /1000] +  ov_B_choiceA + ov_A_choiceA[::-1] + [mean_ov_1B0A/1000]
                cjb_choiceA = [mean_cj_0B1A / 1000] + cjb_B_choiceA + cjb_A_choiceA[::-1] + [mean_cj_1B0A /1000]
                cv_choiceA = [mean_cv_0B1A /1000] + cv_B_choiceA + cv_A_choiceA[::-1] + [mean_cv_1B0A /1000]

            else :
                ov_choiceB = [mean_ov_0B1A / 1000] + ov_B_choiceB + ov_A_choiceB[::-1] + [mean_ov_1B0A/1000]
                cjb_choiceB = [mean_cj_0B1A/1000] + cjb_B_choiceB + cjb_A_choiceB[::-1] + [mean_cj_1B0A /1000]
                cv_choiceB = [mean_cv_0B1A/1000] + cv_B_choiceB + cv_A_choiceB[::-1] + [mean_cv_1B0A/1000]
        print("ov", ov_choiceA, ov_choiceB)
        print("cjb", cjb_choiceA, cjb_choiceB)
        print("cv", cv_choiceA, cv_choiceB)

        """determination of pourcentage of choice B depending on quantity of each juice"""
        pourcentage_choice_B = []
        for i in range(21):
            for j in range(21):
                pourcentage_choice_B.append((choice_B[(i,j)] / self.n) * 100)

        """graphe 4D, H and L"""
        
        nb_4D, nb_4H_A, nb_4H_B=0, 0, 0
        firing_4D, firing_4H_A, firing_4H_B = 0,0,0
        firing_D, firing_H_A, firing_H_B = [], [], []
        for j in range(21):
            for i in range(21):
                for choice_i in self.list_choice:
                    for k in range(2000, 3001):
                        firing_4D += result[(i,j, choice_i)]
                        nb_4D+=1
                    if choice_i =='A':
                        firing_4H_A += result[(i,j, choice_i)]
                        nb_4H_A += 1
                    else :
                        firing_4H_B += result[(i,j, choice_i)]
                        nb_4H_B += 1
                firing_H_A.append(firing_4H_A / nb_4H_A)
                firing_H_B.append(firing_4H_B / nb_4H_B)
            firing_D.append(firing_4D/nb_4D)

        return (ovb_rate_low, ovb_rate_medium, ovb_rate_high, mean_A_chosen_cj, mean_B_chosen_cj,
                mean_low_cv, mean_medium_cv, mean_high_cv,
                ov_choiceA, cjb_choiceA, cv_choiceA, ov_choiceB, cjb_choiceB, cv_choiceB, pourcentage_choice_B)




    def graph(self):
        (ovb_rate_low, ovb_rate_medium, ovb_rate_high, mean_A_chosen_cj, mean_B_chosen_cj, mean_low_cv, mean_medium_cv,
         mean_high_cv, ov_choiceA, cjb_choiceA, cv_choiceA, ov_choiceB, cjb_choiceB, cv_choiceB, pourcentage_choice_B) = self.result_firing_rate()
        """order dictionary before use in graphs"""
        #sort_items = self.ov.keys()
        #self.ov = sorted(sort_items, key=self.compare)
        #sorted les autres !!!!

        X_axis = np.arange(0, self.t_exp, self.dt)

        X2_axis = ["0B: 1A", "1B: 20A", "1B: 16A", "1B: 12A", "1B: 8A", "1B: 4A", "4B: 1A", "8B: 1A", "12B: 1A", "16B: 1A", "20B: 1A", "1B: 0A"]
        bokeh.plotting.output_notebook()

        # figure_3 = bokeh.plotting.figure(title="Figure 3", plot_width=700, plot_height=700)
        figure_4_A = bokeh.plotting.figure(title="Figure 4 A", plot_width=700, plot_height=700)
        figure_4_E = bokeh.plotting.figure(title="Figure 4 E", plot_width=700, plot_height=700)
        figure_4_I = bokeh.plotting.figure(title="Figure 4 I", plot_width=700, plot_height=700)
        figure_4_C = bokeh.plotting.figure(title="Figure 4 C", plot_width=700, plot_height=700)
        figure_4_G = bokeh.plotting.figure(title="Figure 4 G", plot_width=700, plot_height=700)
        figure_4_K = bokeh.plotting.figure(title="Figure 4 K", plot_width=700, plot_height=700)

        # figure_3.circle(x=np.arange(0,20), y=np.arange(0,20), color='blue', size = 10)
        figure_4_A.multi_line([X_axis, X_axis, X_axis], [ovb_rate_low, ovb_rate_medium, ovb_rate_high],
                              color=['red', "green", "blue"])
        figure_4_E.multi_line([X_axis, X_axis], [mean_A_chosen_cj, mean_B_chosen_cj], color=['red', "blue"])
        figure_4_I.multi_line([X_axis, X_axis, X_axis], [mean_low_cv, mean_medium_cv, mean_high_cv],
                              color=['red', "green", "blue"])
        print("ov", ov_choiceA, ov_choiceB )
        print("cjb", cjb_choiceA, cjb_choiceB)
        print("cv", cv_choiceA, cv_choiceB)
        figure_4_C.diamond(x=range(12), y=ov_choiceA, color ='red', size =10)
        figure_4_C.circle(x=range(12), y=ov_choiceB, color = "blue", size =10)
        figure_4_C.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        figure_4_G.diamond(x=range(12), y=cjb_choiceA, color ='red', size = 10)
        figure_4_G.circle(x=range(12), y=cjb_choiceB, color="blue", size=10)
        figure_4_G.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)
        figure_4_K.diamond(x=range(12), y=cv_choiceA, color ='red', size = 10)
        figure_4_K.circle(x=range(12), y=cv_choiceB, color="blue", size=10)
        figure_4_K.circle(x=range(12), y=pourcentage_choice_B, color="black", size=10)


        # bokeh.plotting.show(figure_3)
        bokeh.plotting.show(figure_4_A)
        bokeh.plotting.show(figure_4_E)
        bokeh.plotting.show(figure_4_I)
        bokeh.plotting.show(figure_4_C)
        bokeh.plotting.show(figure_4_G)
        bokeh.plotting.show(figure_4_K)


Class = Economic_Decisions_Model()
Class.graph()