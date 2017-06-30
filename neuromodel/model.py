import numpy as np

from . import history
from .utils import autoinit


class Model:

    @autoinit # set all __init__ arguments as instance members
    def __init__(self,  # Network parameters
                        N_E         = 1600,
                        N_I         = 400,
                        C_ext       = 800,
                        f           = 0.15,
                        r_ext       = 3,               # spike/s

                        # Time constants, synaptic efficacies, and noise
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

                        # Parameters of input-output function for integrate-and-fire neurons
                        I_E = 125,
                        g_E = 0.16,
                        c_E = 310,
                        I_I = 177,
                        g_I = 0.087,
                        c_I = 615,

                        # Parameters used to model OV cells
                        r_o     = 0,                   # spike/s (0 or 6)
                        Δ_r     = 8,                   # spike/s
                        t_offer = 1.0,                 # s
                        a       = 0.175,
                        b       = 0.030,
                        c       = 0.400,
                        d       = 0.100,

                        # Parameters of the experience
                        t_exp = 2.0,                   # s
                        dt    = 0.0005,                # s
                        n     = 4000,                  # number of trials
                        ΔA    = 20,                    # maximum quantity of juice A
                        ΔB    = 20,                    # maximum quantity of juice B

                        # Hebbian learning and synaptic imbalance
                        δ_J_hl   = (1, 1),
                        δ_J_stim = (2, 1),
                        δ_J_gaba = (1, 1, 1),
                        δ_J_nmda = (1, 1),

                        w_p = 1.75,

                        range_A = None,
                        range_B = None,

                        random_seed  = 0,
                        history_keys = ('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb'),
                        full_log     = False, # does trial history log everything?
                        verbose      = False):

        self.random = np.random.RandomState(seed=random_seed)

        self.a = t_offer + a
        self.c = t_offer + c
        self.J_ampa_input = ΔJ * J_ampa_ext_pyr

        self.w_m = 1 - f * (self.w_p - 1) / (1 - self.f)

        # Hebbian learning and synaptic imbalance
        self.δ_J_hl   = {'1': δ_J_hl[0],   '2': δ_J_hl[1]}
        self.δ_J_stim = {'1': δ_J_stim[0], '2': δ_J_stim[1]}
        self.δ_J_nmda = {'1': δ_J_nmda[0], '2': δ_J_nmda[1]}
        self.δ_J_gaba = {'1': δ_J_gaba[0], '2': δ_J_gaba[1], '3': δ_J_gaba[2]}

        # Determination of g maximum for OV cells firing rate
        self.g_max = max(self.g_t(t) for t in np.arange(self.dt, self.t_exp + self.dt, self.dt))

        self.history = history.History(self, keys=history_keys)

        if self.verbose:
            print("Finished model initialization.")


    def firing_rate_pyr(self, i, phi_i): #1
        """Compute the update of the firing rate of pyramidale cells (eq. 1)"""
        return self.dt * ((- self.r[i] + phi_i) / self.τ_ampa)

    def firing_rate_I(self, phi_I):  # 2
        """Compute the update of  the firing rate of interneurons (eq. 2)"""
        return self.dt * ((-self.r['I'] + phi_I) / self.τ_ampa)


    def channel_ampa(self, i):  # 3
        """Compute the update to the AMPA gating variable (eq. 3)"""
        return self.dt * ((-self.S_ampa[i] / self.τ_ampa) + self.r[i])

    def channel_nmda(self, i):  # 4
        """Compute the update to the NMDA gating variable (eq. 4)"""
        return self.dt * ((-self.S_nmda[i] / self.τ_nmda) + (1 - self.S_nmda[i]) * self.γ * self.r[i])

    def channel_gaba(self, i):  # 5
        """Compute the update to the GABA gating variable (eq. 5)"""
        assert i == 'I'
        return self.dt * (-self.S_gaba / self.τ_gaba + self.r[i])


    def Φ(self, I_syn, c, i, gain):  # 6
        """Input-ouput relation for leaky integrate-and-fire cell (Abbott and Chance, 2005) (eq. 6)"""
        return ((c * I_syn - i) / (1 - np.exp(-gain * (c * I_syn - i))))

        ## Currents and parameters

    def I_syn(self, I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim):  # 7
        """Compute the input current for pyramidal cells (eq. 7)"""
        return I_ampa_ext + I_ampa_rec + I_nmda_rec + I_gaba_rec + I_stim

    def I_ampa_ext(self, i):  # 8
        """Compute the external AMPA current for pyramidal cells (eq. 8)"""
        return -self.J_ampa_ext_pyr * self.τ_ampa * self.C_ext * self.r_ext + self.I_eta[i]

    def I_ampa_rec(self, i, j):  # 9
        """Compute the recurrent AMPA current for CJA and CJB cells (eq. 9)"""
        return (-self.N_E * self.f * self.J_ampa_rec_pyr * (self.w_p * self.S_ampa[i] + self.w_m * self.S_ampa[j])
                - self.N_E * (1 - 2 * self.f) * self.J_ampa_rec_pyr * self.w_m * self.S_ampa['3'])

    def I_ampa_rec_3(self):  # 10
        """Compute the recurrent AMPA current for NS cells (eq. 10)"""
        return (-self.N_E * self.f * self.J_ampa_rec_pyr * (self.S_ampa['1'] + self.S_ampa['2'])
                - self.N_E * (1 - 2 * self.f) * self.J_ampa_rec_pyr * self.S_ampa['3'])

    def I_nmda_rec(self, i, j):  # 11
        """Compute the recurrent NMDA current for CJA and CJB cells (eq. 11)"""
        return (-self.N_E * self.f * self.J_nmda_rec_pyr * self.δ_J_nmda[i] * (self.w_p * self.S_nmda[i] + self.w_m * self.S_nmda[j])
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_pyr * self.w_m * self.S_nmda['3'])

    def I_nmda_rec_3(self):  # 12
        """Compute the recurrent NMDA current for NS cells (eq. 12)"""
        return (-self.N_E * self.f * self.J_nmda_rec_pyr * (self.S_nmda['1'] + self.S_nmda['2'])
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_pyr * self.S_nmda['3'])

    def I_gaba_rec(self, i):  # 13
        """Compute the recurrent NMDA current for pyramidal cells (eq. 13)"""
        return -self.N_I * self.J_gaba_rec_pyr * self.δ_J_gaba[i] * self.S_gaba


    def I_ampa_ext_I(self):  # 14
        """Compute the external AMPA current for interneurons (eq. 14)"""
        return -self.J_ampa_ext_in * self.τ_ampa * self.C_ext * self.r_ext + self.I_eta['I']

    def I_ampa_rec_I(self):  # 15
        """Compute the recurrent AMPA current for interneurons (eq. 15)"""
        return (-self.N_E * self.f * self.J_ampa_rec_in * (self.S_ampa['1'] + self.S_ampa['2'])
                - self.N_E * (1 - 2 * self.f) * self.J_ampa_rec_in * self.S_ampa['3'])

    def I_nmda_rec_I(self):  # 16
        """Compute the recurrent NMDA current for interneurons (eq. 16)"""
        return (-self.N_E * self.f * self.J_nmda_rec_in * (self.S_nmda['1'] + self.S_nmda['2'])
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_in * self.S_nmda['3'])

    def I_gaba_rec_I(self):  # 17
        """Compute the recurrent GABA current for interneurons (eq. 17)"""
        return -self.N_I * self.J_gaba_rec_in * self.S_gaba

    def eta(self):
        """Ornstein-Uhlenbeck process (here just Gaussian random noise)"""
        return self.random.normal(0, 1)

    def white_noise(self, j):  # 18
        """Compute the update to I_eta, the noise term (eq. 18)"""
        return self.dt * (-self.I_eta[j]
                          + self.eta() * np.sqrt(self.τ_ampa * self.σ_eta**2) / self.τ_ampa)


    def I_stim(self, i):  # 19
        """Computing the primary input (eq. 19)"""
        assert i in ('1', '2')
        return -self.J_ampa_input * self.δ_J_hl[i] * self.δ_J_stim[i] * self.τ_ampa * self.r_ov[i]

    def g_t(self, t):
        """Computes g_t (eq. 23)"""
        return (1 / (1 + np.exp(- (t - self.a) / self.b))) * (1 / (1 + np.exp((t - self.c) / self.d)))

    def range_adaptation(self, x, x_min, x_max):  # 20
        """Compute the range adaptation of a juice quantity (eq. 20)"""
        return (x - x_min) / (x_max - x_min)

    def firing_ov_cells(self, x, x_min, x_max, t):  # 20, 21, 22, 23
        """Computing the activity profile of OV cells (eq. 20, 21, 22, 23)"""
        x_i = self.range_adaptation(x, x_min, x_max)
        assert(0 <= x_i <= 1)
        g_t = self.g_t(t)

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


    def one_trial(self, x_a, x_b):
        """Compute one trial"""

        # Firing rate of OV B cell, CJ B cell and CV cell for one trial
        self.r      = {'1': 3, '2': 3, '3': 3, 'I': 8}
        self.I_eta  = {'1': 0, '2': 0, '3': 0, 'I': 0}
        self.S_ampa = {'1': 0, '2': 0, '3': 0}
        self.S_nmda = {'1': 0.1, '2': 0.1, '3': 0.1}
        self.S_gaba = 0
        self.choice = None

        self.trial_history = history.TrialHistory(self, x_a, x_b, full_log=self.full_log)

        for t in np.arange(self.dt, self.t_exp + self.dt, self.dt):
            self.one_step(t, x_a, x_b)

        # Determine the final choice in the time window 400-600ms after the offer
        ria, rib = sum(self.trial_history.r_1[2800:3201]), sum(self.trial_history.r_2[2800:3201])
        self.choice = 'B' if ria < rib else 'A'

        self.trial_history.choice = self.choice
        self.history.add_trial(self.trial_history)

    def one_step(self, t, x_a, x_b):
        """Compute one time-step"""
        I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim, I_syn, phi = {}, {}, {}, {}, {}, {}, {}

        # firing rate of ov cells
        self.r_ov = {}  # TODO: do better
        self.r_ov['1'] = self.firing_ov_cells(x_a, self.range_A[0], self.range_A[1], t)
        self.r_ov['2'] = self.firing_ov_cells(x_b, self.range_B[0], self.range_B[1], t)
        # assert r_ova <= 8, 'r_ova = {}'.format(r_ova)
        # assert r_ovb <= 8, 'r_ovb = {}'.format(r_ovb)

        # computing ampa currents
        for i in ['1', '2', '3']:
            I_ampa_ext[i] = self.I_ampa_ext(i)  # equation 8
        I_ampa_ext['I'] = self.I_ampa_ext_I()  # equation 14

        for i, j in [('1', '2'), ('2', '1')]:  # i != j
            I_ampa_rec[i] = self.I_ampa_rec(i, j)  # equation 9
        I_ampa_rec['3'] = self.I_ampa_rec_3()  # equation 12
        I_ampa_rec['I'] = self.I_ampa_rec_I()  # equation 15

        # computing nmda currents
        for i, j in [('1', '2'), ('2', '1')]:  # i != j
            I_nmda_rec[i] = self.I_nmda_rec(i, j)  # equation 10
        I_nmda_rec['3'] = self.I_nmda_rec_3()  # equation 11
        I_nmda_rec['I'] = self.I_nmda_rec_I()  # equation 16

        # computing gaba currents
        for i in ['1', '2', '3']:
            I_gaba_rec[i] = self.I_gaba_rec(i)  # equation 13
        I_gaba_rec['I'] = self.I_gaba_rec_I()  # equation 17

        # computing primary input currents
        for i in ['1', '2']:
            I_stim[i] = self.I_stim(i)  # equation 19
        for i in ['3', 'I']:
            I_stim[i] = 0

        # computing the input currents
        for i in ['1', '2', '3', 'I']:
            I_syn[i] = self.I_syn(I_ampa_ext[i], I_ampa_rec[i], I_nmda_rec[i], I_gaba_rec[i], I_stim[i])  # equation 7

        # updating gating variables
        for i in ['1', '2', '3']:
            self.S_ampa[i] += self.channel_ampa(i)  # equation 3
            self.S_nmda[i] += self.channel_nmda(i)  # equation 4
        self.S_gaba += self.channel_gaba('I')  # equation 5

        for i in ['1', '2', '3']:
            phi[i] = self.Φ(I_syn[i], self.c_E, self.I_E, self.g_E)  # equation 6
        phi['I'] = self.Φ(I_syn['I'], self.c_I, self.I_I, self.g_I)  # equation 6

        for i in ['1', '2', '3']:
            self.r[i] += self.firing_rate_pyr(i, phi[i])  # equation 1
        self.r['I'] += self.firing_rate_I(phi['I'])  # equation 2

        # generating noise
        for j in ['1', '2', '3', 'I']:
            self.I_eta[j] += self.white_noise(j)  # equation 18


        self.trial_history.update(self, I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim, I_syn, phi)
