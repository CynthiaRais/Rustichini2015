#-*- coding: utf-8 -*-

import numpy as np
import math
import collections # to order items in dictionary result
import bokeh
import bokeh.plotting

np.random.seed(100)

def firing_pyr_cells(r_i, phi, τ_ampa, dt): #1
    """Update the firing rate of pyramidale cells (eq. 1)"""
    r_i += ((- r_i + phi) / τ_ampa) * dt
    #assert 0 <= r_i <= 30, " r_i = {0}, phi = {1}".format(r_i, phi)
    return r_i

def firing_rate_I(r_I, phi, τ_gaba, dt): #2
    """Update the firing rate of interneurons (eq. 2)"""
    r_I += ((-r_I + phi) / τ_gaba) * dt
    #assert 10 <= r_I <= 20, "r_I = {0}".format(r_I)
    return r_I

def channel_ampa(S_ampa, τ_ampa, r_i, dt): #3
    """Open AMPA channels (eq. 3)"""
    S_ampa += ((- S_ampa / τ_ampa) + r_i) * dt
    #assert 0 <= S_ampa <= 1, "S_ampa = {0}".format(S_ampa)
    return S_ampa

def channel_nmda(S_nmda, τ_nmda, γ, r_i, dt): #4
    """Open NMDA channels (eq. 4)"""
    S_nmda += ((-S_nmda / τ_nmda) + (1 - S_nmda) * γ * r_i) * dt
    #assert 0 <= S_nmda <= 1, "S_nmda = {0}".format(S_nmda)
    return S_nmda

def channel_gaba(S_gaba, τ_gaba, r_I, dt): #5
    """Open GABA channels (eq. 5)"""
    S_gaba += (-S_gaba / τ_gaba + r_I) * dt
    #assert 0 <= S_gaba <= 1, "S_gaba = {0}".format(S_gaba)
    return S_gaba

def Φ(I_syn, c, gain, i): # 6
    """Input-ouput relation for leaky integrate-and-fire cell (Abbott and Chance, 2005) (eq. 6)"""
    phi = ((c * I_syn - i) / (1 - np.exp(-gain * (c * I_syn - i))))
    return phi


    ## Currents and parameters

def I_syn(I_ampa_ext, I_ampa_rec, I_nmda_rec, I_gaba_rec, I_stim): #7
    """Compute the input current for pyramidal cells (eq. 7)"""
    return I_ampa_ext + I_ampa_rec + I_nmda_rec + I_gaba_rec + I_stim

def I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta): #8
    """Compute the external AMPA current for pyramidal cells (eq. 8)"""
    return -J_ampa_ext_pyr * τ_ampa * C_ext * r_ext + I_eta

def I_ampa_rec(N_E, f, J_ampa_pyr, w_p, w_m, S_ampa_1, S_ampa_2, S_ampa_3): #9
    """Compute the recurrent AMPA current for CJA and CJB cells (eq. 9)"""
    return (-N_E * f * J_ampa_pyr * (w_p * S_ampa_1 + w_m * S_ampa_2)
            - N_E * (1 - 2 * f) * J_ampa_pyr * w_m * S_ampa_3)

def I_ampa_rec_3(N_E, f, J_ampa_pyr, S_ampa_1, S_ampa_2, S_ampa_3): #10
    """Compute the recurrent AMPA current for NS cells (eq. 10)"""
    return (-N_E * f * J_ampa_pyr * (S_ampa_1 + S_ampa_2)
            - N_E * (1 - 2 * f) * J_ampa_pyr * S_ampa_3)

def I_nmda_rec(N_E, f, J_nmda_pyr, δ_j_ndma, w_p, S_nmda_1, w_m, S_nmda_2, S_nmda_3): #11
    """Compute the recurrent NMDA current for CJA and CJB cells (eq. 11)"""
    return (-N_E * f * J_nmda_pyr * δ_j_ndma * (w_p * S_nmda_1 + w_m * S_nmda_2)
            - N_E * (1 - 2 * f) * J_nmda_pyr * w_m * S_nmda_3)

def I_nmda_rec_3(N_E, f, J_nmda_pyr, S_nmda_1, S_nmda_2, S_nmda_3): #12
    """Compute the recurrent NMDA current for NS cells (eq. 12)"""
    return(-N_E * f * J_nmda_pyr * (S_nmda_1 + S_nmda_2)
           - N_E *(1 - 2 * f) * J_nmda_pyr * S_nmda_3)

def I_gaba_rec(N_I, J_gaba_pyr, δ_J_gaba, S_gaba): #13
    """Compute the recurrent NMDA current for pyramidal cells (eq. 13)"""
    return -N_I * J_gaba_pyr * δ_J_gaba * S_gaba

def I_ampa_ext_I(J_ampa_ext_in, τ_ampa, C_ext, r_ext, I_eta): #14
    """Compute the external AMPA current for interneurons (eq. 14)"""
    return -J_ampa_ext_in * τ_ampa * C_ext * r_ext + I_eta

def I_ampa_rec_I(N_E, f, J_ampa_rec_in, S_ampa_1, S_ampa_2, S_ampa_3): #15
    """Compute the recurrent AMPA current for interneurons (eq. 15)"""
    return (-N_E * f * J_ampa_rec_in * (S_ampa_1 + S_ampa_2)
            - N_E * (1 - 2 * f) * J_ampa_rec_in * S_ampa_3)

def I_nmda_rec_I(N_E, f, J_nmda_rec_in, S_nmda_1, S_nmda_2, S_nmda_3) : #16
    """Compute the recurrent NMDA current for interneurons (eq. 16)"""
    return (-N_E * f * J_nmda_rec_in * (S_nmda_1 + S_nmda_2)
            - N_E * (1 - 2 * f) * J_nmda_rec_in * S_nmda_3)

def I_gaba_rec_I(N_I, J_gaba_rec_in, S_gaba): #17
    """Compute the recurrent GABA current for interneurons (eq. 17)"""
    return -N_I * J_gaba_rec_in * S_gaba

def eta():
    """Ornstein-Uhlenbeck process (here just Gaussian random noise)"""
    return np.random.normal(0, 1)


def white_noise(I_eta, τ_ampa, σ_eta, dt): #18
    """Update I_eta, the noise term (eq. 18)"""
    I_eta += ((-I_eta + eta() * math.sqrt(τ_ampa * (σ_eta ** 2))) / τ_ampa) * dt
    return I_eta

def I_stim(J_ampa_input, δ_j_hl, δ_j_stim, τ_ampa, r_ov): #19
    """Computng the primary input (eq. 19)"""
    return -J_ampa_input * δ_j_hl * δ_j_stim * τ_ampa * r_ov

def firing_ov_cells(x, xmin, x_max, t, r_o, Δ_r): #20, 21, 22, 23
    """Computing the activity profile of OV cells (eq. 20, 21, 22, 23)"""
    x_i = (x - xmin) / (x_max - xmin)
    g_t = (1 / (1 + np.exp(- (t - a) / b))) * (1 / (1 + np.exp((t - c) / d)))
    assert 0 <= g_t <= 1, "g_t == {0}".format(g_t)
    f_t = g_t / g_max
    r_ov = r_o + Δ_r * f_t * x_i
    assert 0 <= f_t <= 1, "f_t = {0}, g_t = {1}, g_max1 = {2}, t = {3}".format(f_t, g_t, g_max, t)
    assert 0 <= r_ov <= 8, "r_ov = {0}".format(r_ov)
    return r_ov


    ## Parameters used in simulation (Table 1)

# Network parameters
N_E            = 1600
N_I            = 400
C_ext          = 800
f              = 0.15
r_ext          = 3 # spike/s

# Time constants, synaptic efficacies, and noise
τ_ampa         = 0.002 # s
τ_nmda         = 0.100 # s
τ_gaba         = 0.005 # s
J_ampa_ext_pyr = -0.1123
J_ampa_rec_pyr = -0.0027
J_nmda_rec_pyr = -0.00091979
J_gaba_rec_pyr = 0.0215
J_ampa_ext_in  = -0.0842
J_ampa_rec_in  = -0.0022
J_nmda_rec_in  = -0.00083446
J_gaba_rec_in  = 0.0180
γ              = 0.641
σ_eta          = 0.020

# Parameters of input-output function for integrate-and-fire neurons
I_E = 125
g_E = 0.16
c_E = 310
I_I = 177
g_I = 0.087
c_I = 615

# Parameters used to model OV cells
r_o = 0 # spike/s (0 or 6)
Δ_r = 8 # spike/s
t_offer = 0.5 # s
a = t_offer + 0.175 # s
b = 0.030 # s
c = t_offer + 0.400 # s
d = 0.100 # s
J_ampa_input = 30 * J_ampa_ext_pyr

dt = 0.0005 # s


    ##

g_max = 0
list_g = []

t=0


while t < 1.500:
    g = (1 / (1 + np.exp(- (t- a) / b))) * (1 / (1 + np.exp((t - c) / d)))
    list_g.append(g)
    t+=dt
g_max = np.max(list_g)






w_p = 1.75
w_m = 1 - f * (w_p - 1) / (1 - f)




δ_j_hl_cj_a, δ_j_hl_cj_b = 1, 1
δ_j_stim_cj_a, δ_j_stim_cj_b = 2, 1
δ_J_gaba_cj_a, δ_J_gaba_cj_b, δ_J_gaba_ns = 1, 1, 1
δ_J_nmda_cj_a, δ_J_nmda_cj_b = 1, 1

I_stim_cv, I_stim_ns = 0, 0


def ov_a_cells(t, x_a, xmin, x_max):
    r_ov_a = firing_ov_cells(x_a, xmin, x_max, t, r_o, Δ_r)
    return r_ov_a


def ov_b_cells(t, x_b, xmin, x_max):
    r_ov_b = firing_ov_cells(x_b, xmin, x_max, t, r_o, Δ_r)
    return r_ov_b


def cj_a_cells(t, r_i_cj_a, S_ampa_cj_a, S_nmda_cj_a, S_gaba_cj_a, I_eta_cj_a,
               S_ampa_cj_b, S_ampa_ns, S_nmda_cj_b, S_nmda_ns, r_i_cv_cells, r_ov_a):

    S_ampa_cj_a = channel_ampa(S_ampa_cj_a, τ_ampa, r_i_cj_a, dt) #equation 3
    S_nmda_cj_a = channel_nmda(S_nmda_cj_a, τ_nmda, γ, r_i_cj_a, dt) #equation 4
    S_gaba_cj_a = channel_gaba(S_gaba_cj_a, τ_gaba, r_i_cv_cells, dt) #equation 5
    S_cj_a = [S_ampa_cj_a, S_nmda_cj_a, S_gaba_cj_a]

    I_eta_cj_a = white_noise(I_eta_cj_a, τ_ampa, σ_eta, dt) #equation 18
    I_ampa_ext_cj_a = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_cj_a) #equation 8
    I_ampa_rec_cj_a = I_ampa_rec(N_E, f, J_ampa_rec_pyr, w_p, w_m, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns) #equation 9
    I_nmda_rec_cj_a = I_nmda_rec(N_E, f, J_nmda_rec_pyr, δ_J_nmda_cj_a, w_p, S_nmda_cj_a, w_m, S_nmda_cj_b, S_nmda_ns) #equation 11
    I_gaba_rec_cj_a = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_cj_a, S_gaba_cj_a) #equation 13
    I_stim_cj_a = I_stim(J_ampa_input, δ_j_hl_cj_a, δ_j_stim_cj_a, τ_ampa, r_ov_a) #equation 19
    I_syn_cj_a = I_syn(I_ampa_ext_cj_a, I_ampa_rec_cj_a, I_nmda_rec_cj_a, I_gaba_rec_cj_a, I_stim_cj_a) #equation 7
    #print("I_syn_cj_a = I_ampa_ext_cj_a + I_nmda_rec_cj_a + I_gaba_rec_cj_a + I_stim_cj_a", I_syn_cj_a, I_ampa_ext_cj_a, I_ampa_rec_cj_a, I_nmda_rec_cj_a , I_gaba_rec_cj_a, I_stim_cj_a)

    phi_cj_a = Φ(I_syn_cj_a, c_E, g_E, I_E) #equation 6
    r_i_cj_a = firing_pyr_cells(r_i_cj_a, phi_cj_a, τ_gaba, dt) #equation 1
    return r_i_cj_a, S_cj_a, I_eta_cj_a, I_ampa_ext_cj_a, I_nmda_rec_cj_a, I_gaba_rec_cj_a, I_stim_cj_a


def cj_b_cells(t, r_i_cj_b, S_ampa_cj_b, S_nmda_cj_b, S_gaba_cj_b, I_eta_cj_b,
               S_ampa_cj_a, S_ampa_ns, S_nmda_cj_a, S_nmda_ns, r_i_cv_cells, r_ov_b):

    S_ampa_cj_b = channel_ampa(S_ampa_cj_b, τ_ampa, r_i_cj_b, dt) # equation 3
    S_nmda_cj_b = channel_nmda(S_nmda_cj_b, τ_nmda, γ, r_i_cj_b, dt) # equation 4
    S_gaba_cj_b = channel_gaba(S_gaba_cj_b, τ_gaba, r_i_cv_cells, dt) # equation 5
    S_cj_b = [S_ampa_cj_b, S_nmda_cj_b, S_gaba_cj_b]

    I_eta_cj_b = white_noise(I_eta_cj_b, τ_ampa, σ_eta, dt) # equation 18
    I_ampa_ext_cj_b = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_cj_b) # equation 8
    I_ampa_rec_cj_b = I_ampa_rec(N_E, f, J_ampa_rec_pyr, w_p, w_m, S_ampa_cj_b, S_ampa_cj_a, S_ampa_ns) # equation 9
    I_nmda_rec_cj_b = I_nmda_rec(N_E, f, J_nmda_rec_pyr, δ_J_nmda_cj_b, w_p, S_nmda_cj_b, w_m, S_nmda_cj_a, S_nmda_ns) # equation 11
    I_gaba_rec_cj_b = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_cj_b, S_gaba_cj_b) # equation 13
    I_stim_cj_b = I_stim(J_ampa_input, δ_j_hl_cj_b, δ_j_stim_cj_b, τ_ampa, r_ov_b) # equation 19
    I_syn_cj_b = I_syn(I_ampa_ext_cj_b, I_ampa_rec_cj_b, I_nmda_rec_cj_b, I_gaba_rec_cj_b, I_stim_cj_b) #equation 7
    #print("I_syn_cj_b = I_ampa_ext_cj_b + I_nmda_rec_cj_b + I_gaba_rec_cj_b + I_stim_cj_b", I_syn_cj_b, I_ampa_ext_cj_b,
          #I_ampa_rec_cj_b, I_nmda_rec_cj_b, I_gaba_rec_cj_b, I_stim_cj_b)

    phi_cj_b = Φ(I_syn_cj_b, c_E, g_E, I_E) #equation 6
    r_i_cj_b = firing_pyr_cells(r_i_cj_b, phi_cj_b, τ_ampa, dt) #equation 1
    #assert 0 < r_i_cj_b <= 30, "r_i_cj_b = {0}".format(r_i_cj_b)
    return r_i_cj_b, S_cj_b, I_eta_cj_b, I_syn_cj_b, I_stim_cj_b, I_ampa_ext_cj_b


def ns_cells(t, r_i_ns, S_ampa_ns, S_nmda_ns, S_gaba_ns, I_eta_ns,
             S_ampa_cj_a, S_ampa_cj_b, S_nmda_cj_a, S_nmda_cj_b, r_i_cv_cells):

    S_ampa_ns = channel_ampa(S_ampa_ns, τ_ampa, r_i_ns, dt) #equation 3
    S_nmda_ns = channel_nmda(S_nmda_ns, τ_nmda, γ, r_i_ns, dt) #equation 4
    S_gaba_ns = channel_gaba(S_gaba_ns, τ_gaba, r_i_cv_cells, dt) #equation 5
    S_ns = [S_ampa_ns, S_nmda_ns, S_gaba_ns]

    I_eta_ns = white_noise(I_eta_ns, τ_ampa, σ_eta, dt) #equation 18
    I_ampa_ext_ns = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_ns) #equation 8
    I_ampa_rec_ns = I_ampa_rec_3(N_E, f, J_ampa_rec_pyr, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns) #equation 10
    I_nmda_rec_ns = I_nmda_rec_3(N_E, f, J_nmda_rec_pyr, S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns) #equation 12
    I_gaba_rec_ns = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_ns, S_gaba_ns) #equation 13
    I_stim_ns = 0
    I_syn_ns = I_syn(I_ampa_ext_ns, I_ampa_rec_ns, I_nmda_rec_ns, I_gaba_rec_ns, I_stim_ns) #equation 7

    phi_ns = Φ(I_syn_ns, c_E, g_E, I_E) #equation 6
    r_i_ns = firing_pyr_cells(r_i_ns, phi_ns, τ_ampa, dt) #equation 1
    return r_i_ns, S_ns, I_eta_ns, I_syn_ns



def cv_cells(t, r_i_cv_cells, S_gaba_cv, I_eta_cv,
             S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns,
             S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns):

    S_gaba_cv = channel_gaba(S_gaba_cv, τ_gaba, r_i_cv_cells, dt) #equation 5

    I_eta_cv = white_noise(I_eta_cv, τ_ampa, σ_eta, dt)  #equation 18
    I_ampa_ext_cv = I_ampa_ext_I(J_ampa_ext_in, τ_ampa, C_ext, r_ext, I_eta_cv)  #equation 14
    I_ampa_rec_cv = I_ampa_rec_I(N_E, f, J_ampa_rec_in, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns) #equation 15
    I_nmda_rec_cv = I_nmda_rec_I(N_E, f, J_nmda_rec_in, S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns) #equation 16
    I_gaba_rec_cv = I_gaba_rec_I(N_I, J_gaba_rec_in, S_gaba_cv) #equation 17
    I_syn_cv_cells = I_syn(I_ampa_ext_cv, I_ampa_rec_cv, I_nmda_rec_cv, I_gaba_rec_cv, I_stim_cv) #equation 7

    phi_cv_cells = Φ(I_syn_cv_cells, c_I, g_I, I_I) #equation 6
    #assert phi_cv_cells == np.nan, "phi_cv_cells = {0}, I_ampa_ext_cv  = {1}, I_nmda_rec_cv = {2}, I_gaba_rec_cv = {3}, I_syn_cv_cells  = {4}".format(phi_cv_cells, I_ampa_ext_cv, I_nmda_rec_cv, I_gaba_rec_cv, I_syn_cv_cells  )
    r_i_cv_cells = firing_rate_I(r_i_cv_cells, phi_cv_cells, τ_gaba, dt) #equation 2
    return r_i_cv_cells, S_gaba_cv, I_eta_cv, I_syn_cv_cells


def quantity_juice():
    ''' random choice of juice quantity '''

    x_a = np.random.randint(0, 20)
    x_b = np.random.randint(0, 20)
    while x_a == 0 and x_b == 0:
        p = np.random.random()
        if p < 0.5:
            x_a = np.random.randint(0, 20)
        else:
            x_b = np.random.randint(0, 20)
    return x_a, x_b


def one_trial(x_a, x_b, xmin_list, x_max_list, t,
              r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
              I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
              S_cj_a, S_cj_b, S_ns, S_gaba_cv):

    mean_ov_b, r_i_cj_b_tot, r_i_cj_a_tot, r_i_ns_tot, r_i_cv_cells_tot = [], [], [], [], []
    s_ampa_cj_b_tot, s_nmda_b_tot, s_gaba_b_tot, I_eta_b_tot = [], [], [], []
    s_gaba_cv_tot, I_eta_cv_tot = [], []
    i_ampa_ext_cj_b_tot, i_syn_cj_b_tot, i_ampa_rec_cj_b_tot, i_syn_cv_tot = [], [], [], []
    i_nmda_cj_b_tot, i_gaba_cj_b_tot, i_stim_cj_b_tot, i_eta_cj_b_tot = [], [], [], []
    i_ampa_ext_cj_a_tot = []
    for t in np.arange(0, 1.5, 0.0005):

        r_ov_a = ov_a_cells(t, x_a, xmin_list[0], x_max_list[0])

        r_ov_b = ov_b_cells(t, x_b, xmin_list[1], x_max_list[1])

        r_i_cj_a, S_cj_a, I_eta_cj_a, I_ampa_ext_cj_a, I_nmda_rec_cj_a, I_gaba_rec_cj_a, I_stim_cj_a = cj_a_cells(t, r_i_cj_a, S_cj_a[0], S_cj_a[1], S_cj_a[2],
                                                 I_eta_cj_a, S_cj_b[0], S_ns[0], S_cj_b[1], S_ns[1], r_i_cv_cells, r_ov_a)
        r_i_cj_b, S_cj_b, I_eta_cj_b, I_syn_cj_b, I_stim_cj_b, I_ampa_ext_cj_b= cj_b_cells(t, r_i_cj_b, S_cj_b[0], S_cj_b[1], S_cj_b[2],
                                                 I_eta_cj_b, S_cj_a[0], S_ns[0], S_cj_a[1], S_ns[1], r_i_cv_cells, r_ov_b)

        r_i_ns, S_ns, I_eta_ns, I_syn_ns = ns_cells(t, r_i_ns, S_ns[0], S_ns[1], S_ns[2],
                                         I_eta_ns,S_cj_a[0], S_cj_b[0], S_cj_a[1], S_cj_b[1], r_i_cv_cells)

        r_i_cv_cells, S_gaba_cv, I_eta_cv, I_syn_cv_cells = cv_cells(t, r_i_cv_cells, S_gaba_cv,
                                                    I_eta_cv, S_cj_a[0], S_cj_b[0], S_ns[0], S_cj_a[1], S_cj_b[1], S_ns[1])

        #print( "r_i_cv_cells = {0}".format(r_i_cv_cells))

        #return (r_ov_a , r_ov_b, r_i_cj_a, S_cj_a, r_i_cj_b, S_cj_b, r_i_ns, S_ns, r_i_cv_cells, S_gaba_cv)



        mean_ov_b.append(r_ov_b)
        r_i_cj_b_tot.append(r_i_cj_b)
        r_i_cj_a_tot.append(r_i_cj_a)
        r_i_ns_tot.append(r_i_ns)
        r_i_cv_cells_tot.append(r_i_cv_cells)
        s_ampa_cj_b_tot.append(S_cj_b[0])
        s_nmda_b_tot.append(S_cj_b[1])
        s_gaba_b_tot.append(S_cj_b[2])
        s_gaba_cv_tot.append(S_gaba_cv)
        i_stim_cj_b_tot.append(I_stim_cj_b)
        i_ampa_ext_cj_b_tot.append(I_ampa_ext_cj_b)
        i_ampa_ext_cj_a_tot.append(I_ampa_ext_cj_a)



    #return (mean_ov_b, r_i_cj_a_tot, r_i_cj_b_tot, r_i_ns_tot, s_ampa_cj_b_tot, s_nmda_b_tot, r_i_cv_cells_tot, s_gaba_cv_tot,
    #        i_ampa_ext_cj_a_tot, i_nmda_cj_a_tot, i_gaba_cj_a_tot, i_stim_cj_a_tot, i_syn_cj_b_tot, i_syn_ns_tot, i_syn_cv_tot)

    if r_i_cj_a > r_i_cj_b :
        choice = 'choice A'
    elif r_i_cj_a < r_i_cj_b:
        choice = 'choice B'
    else :
        raise ValueError(choice ='no choice')

    return (choice, mean_ov_b, r_i_cj_b_tot, r_i_cv_cells_tot, i_stim_cj_b_tot, i_ampa_ext_cj_b_tot, i_ampa_ext_cj_a_tot)

    #print(choice, np.max(mean_ov_b), np.max(r_i_cj_a_tot), np.max(r_i_cj_b_tot), np.max(r_i_ns_tot), np.max(r_i_cv_cells_tot), np.max(s_ampa_cj_b_tot), np.max(s_nmda_b_tot), np.max(s_gaba_b_tot), np.max(s_gaba_cv_tot),
    #       np.min(mean_ov_b), np.min(r_i_cj_a_tot), np.min(r_i_cj_b_tot), np.min(r_i_ns_tot), np.min(r_i_cv_cells_tot),
    #       np.min(s_ampa_cj_b_tot), np.min(s_nmda_b_tot), np.min(s_gaba_b_tot), np.min(s_gaba_cv_tot))
    #return (choice, np.max(mean_ov_b), np.max(r_i_cj_a_tot), np.max(r_i_cj_b_tot), np.max(r_i_ns_tot), np.max(r_i_cv_cells_tot), np.max(s_ampa_cj_b_tot), np.max(s_nmda_b_tot), np.max(s_gaba_b_tot), np.max(s_gaba_cv_tot),
    #       np.min(mean_ov_b), np.min(r_i_cj_a_tot), np.min(r_i_cj_b_tot), np.min(r_i_ns_tot), np.min(r_i_cv_cells_tot),
    #       np.min(s_ampa_cj_b_tot), np.min(s_nmda_b_tot), np.min(s_gaba_b_tot), np.min(s_gaba_cv_tot))


#(mean_ov_b, r_i_cj_a_tot, r_i_cj_b_tot, r_i_ns_tot, s_ampa_cj_b_tot, s_nmda_b_tot, r_i_cv_cells_tot, s_gaba_cv_tot,
# i_ampa_ext_cj_a_tot, i_nmda_cj_a_tot, i_gaba_cj_a_tot, i_stim_cj_a_tot, i_syn_cj_b_tot, i_syn_ns_tot, i_syn_cv_tot) = one_trial(20, 10, [0, 0], [20, 20], 0, 0, 0, 0, 0,
#               0,0,0,0, [0, 0, 0], [ 0, 0, 0], [0, 0, 0], 0)

#print(r_i_cj_a_tot[0], r_i_cj_a_tot[1], r_i_cj_a_tot[2])

#bokeh.plotting.output_notebook()
#X_axis = range(len(r_i_cj_b_tot))
#X2_axis = range(len(mean_ov_b))
#figure_4_A = bokeh.plotting.figure(title="ovb", plot_width=300, plot_height=300)
#figure_4_E = bokeh.plotting.figure(title="r_i_cj_a", plot_width=300, plot_height=300)
#figure_4_I = bokeh.plotting.figure(title="r_i_b", plot_width=300, plot_height=300)
#figure_4_ns = bokeh.plotting.figure(title="i_ampa_ext_cj_a_tot", plot_width=300, plot_height=300)
#figure_4_cv = bokeh.plotting.figure(title="ri_cv", plot_width=300, plot_height=300)
#fig_i_syn_a = bokeh.plotting.figure(title="i_nmda_cj_a_tot", plot_width=300, plot_height=300)
#fig_i_syn_b = bokeh.plotting.figure(title="i_syn_cj_b", plot_width=300, plot_height=300)
#fig_i_syn_ns = bokeh.plotting.figure(title="i_syn_ns", plot_width=300, plot_height=300)
#fig_i_syn_cv = bokeh.plotting.figure(title="i_syn_cv", plot_width=300, plot_height=300)
#figure_4_A.line(X2_axis , mean_ov_b, color ='red')
#figure_4_E.line(X_axis , r_i_cj_a_tot, color ='red')
#figure_4_I.line(X_axis , r_i_cj_b_tot, color ='red')
#figure_4_ns.line(X_axis, i_ampa_ext_cj_a_tot, color = 'red')
#figure_4_cv.line(X_axis, r_i_cv_cells_tot, color = 'red')
#fig_i_syn_a.line(X_axis, i_nmda_cj_a_tot, color = 'green')
#fig_i_syn_b.line(X_axis, i_syn_cj_b_tot, color = 'green')
#fig_i_syn_ns.line(X_axis, i_syn_ns_tot, color = 'green')
#fig_i_syn_cv.line(X_axis, i_syn_cv_tot, color = 'green')
#graph_ampa = bokeh.plotting.figure(title="i_gaba_cj_a_tot", plot_width=300, plot_height=300)
#graph_nmda = bokeh.plotting.figure(title="i_stim_cj_a_tot", plot_width=300, plot_height=300)

#graph_gaba_cv = bokeh.plotting.figure(title="gaba cv b", plot_width=300, plot_height=300)
#graph_ampa.line(X_axis , i_gaba_cj_a_tot, color ='blue')
#graph_nmda.line(X_axis , i_stim_cj_a_tot, color ='blue')

#graph_gaba_cv.line(X_axis , s_gaba_cv_tot, color ='blue')

#bokeh.plotting.show(figure_4_A)
#bokeh.plotting.show(figure_4_E)
#bokeh.plotting.show(figure_4_I)
#bokeh.plotting.show(figure_4_ns)
#bokeh.plotting.show(graph_ampa)
#bokeh.plotting.show(graph_nmda)
#bokeh.plotting.show(figure_4_cv)
#bokeh.plotting.show(graph_gaba_cv)
#bokeh.plotting.show(fig_i_syn_a)
#bokeh.plotting.show(fig_i_syn_b)
#bokeh.plotting.show(fig_i_syn_ns)
#bokeh.plotting.show(fig_i_syn_cv)


def session():
    r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells = 0, 0, 0, 0
    I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv = 0, 0, 0, 0
    S_cj_a, S_cj_b, S_ns = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    S_gaba_cv = 0
    t = 0
    quantity_a, quantity_b = [], []

    result = {}

    for j in range(0, 21):
        for k in range(0, 21):
                result[(j, k)] = []
    result = collections.OrderedDict(sorted(result.items(), key = lambda t: t[0]))
    for i in range(100):
        x_a, x_b = quantity_juice()
        quantity_a.append(x_a)
        quantity_b.append(x_b)

    xmin_a = np.min(quantity_a)
    xmin_b = np.min(quantity_b)
    x_max_a = np.max(quantity_a)
    x_max_b = np.max(quantity_b)
    xmin_list = [xmin_a, xmin_b]
    x_max_list = [x_max_a, x_max_b]

    for i in range(100):
        (choice, mean_ov_b, r_i_cj_b_tot, r_i_cv_cells_tot, i_stim_cj_b_tot, i_ampa_ext_cj_b_tot, i_ampa_ext_cj_a_tot) = one_trial(quantity_a[i], quantity_b[i], xmin_list, x_max_list, t,
                                                        r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
                                                        I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
                                                        S_cj_a, S_cj_b, S_ns, S_gaba_cv)

        result[(quantity_a[i], quantity_b[i])].append([choice, mean_ov_b, r_i_cj_b_tot, r_i_cv_cells_tot,
                                                       i_stim_cj_b_tot, i_ampa_ext_cj_b_tot, i_ampa_ext_cj_a_tot])
    return result



def result_firing_rate():

    ''' on obtient la moyenne des ov_b rate en fonction du temps
    et si l'essai a eu une offre forte, moyenne, faible '''

    ovb_rate_low, ovb_rate_high, ovb_rate_medium = [], [], []

    result = session()
    mean_A_chosen_cj, mean_B_chosen_cj = [], []
    mean_i_stim_choice_a, mean_i_stim_choice_b = [], []
    mean_low_cv, mean_medium_cv, mean_high_cv = [], [], []
    mean_i_ampa_ext_choice_a, mean_i_ampa_ext_choice_b, mean_i_ampa_rec_choice_a, mean_i_ampa_rec_choice_b = [], [], [], []
    mean_i_ampa_ext_a_choice_a, mean_i_ampa_ext_a_choice_b, mean_i_gaba_choice_a, mean_i_gaba_choice_b = [], [], [], []
    mean_i_eta_choice_a, mean_i_eta_choice_b = [], []
    ''' le terme k représente le temps,
     le terme i représente la quantité de A,
      le j représente la quantité de B
      et le l représente la liste de l'essai l
      pour un temps donné, on ajoute les r_ov_b pour chaque (i,j) pour chaque essai l '''

    for k in range(3000):
        mean_ov_low, mean_ov_high, mean_ov_medium = 0, 0, 0
        low, medium, high = 0, 0, 0
        for i in range(21):
            for j in range(6):
                for l in range(len(result[(i,j)])):
                     mean_ov_low += result[(i,j)][l][1][k]
                     low += 1
            for j in range(7,13):
                    for l in range(len(result[(i,j)])):
                        mean_ov_medium += result[(i,j)][l][1][k]
                        medium += 1
            for j in range(14,21):
                    for l in range(len(result[(i,j)])):
                        mean_ov_high += result[(i,j)][l][1][k]
                        high += 1
        ovb_rate_low.append(mean_ov_low / low)
        ovb_rate_medium.append(mean_ov_medium / medium)
        ovb_rate_high.append(mean_ov_high / high)

    for k in range (3000):
        A_chosen_cj, B_chosen_cj = 0, 0
        chosen_value_low, chosen_value_medium, chosen_value_high = 0, 0, 0
        A_nb, B_nb = 0, 0
        low_cv, medium_cv, high_cv = 0, 0, 0
        i_stim_choice_a, i_stim_choice_b = 0, 0
        i_ampa_ext_choice_a, i_ampa_ext_a_choice_a, i_nmda_choice_a, i_gaba_choice_a = 0, 0, 0, 0
        i_ampa_ext_choice_b, i_ampa_ext_a_choice_b, i_nmda_choice_b, i_gaba_choice_b = 0, 0, 0, 0
        i_eta_choice_a, i_eta_choice_b = 0, 0
        for i in range(21):
            for j in range(21):
                for l in range(len(result[(i,j)])):
                    if result[(i, j)][l][0] == 'choice A':
                        A_chosen_cj += result[(i, j)][l][2][k]
                        A_nb +=1
                        i_stim_choice_a += result[(i, j)][l][4][k]
                        i_ampa_ext_choice_a += result[(i, j)][l][5][k]
                        i_ampa_ext_a_choice_a += result[(i, j)][l][6][k]
                        if i < 6 :
                            chosen_value_low += result[(i, j)][l][3][k]
                            low_cv +=1
                        elif 6 < i < 13:
                            chosen_value_medium += result[(i,j)][l][3][k]
                            medium_cv += 1
                        else :
                            chosen_value_high += result[(i, j)][l][3][k]
                            high_cv += 1
                    else :
                        B_chosen_cj += result[(i, j)][l][2][k]
                        B_nb +=1
                        i_stim_choice_b += result[(i, j)][l][4][k]
                        i_ampa_ext_choice_b += result[(i, j)][l][5][k]
                        i_ampa_ext_choice_b += result[(i, j)][l][6][k]
                        if j < 6 :
                            chosen_value_low += result[(i, j)][l][3][k]
                            low_cv +=1
                        elif 6 < j < 13:
                            chosen_value_medium += result[(i, j)][l][3][k]
                            medium_cv +=1
                        else :
                            chosen_value_high += result[(i, j)][l][3][k]
                            high_cv +=1

        mean_A_chosen_cj.append(A_chosen_cj / A_nb)
        mean_B_chosen_cj.append(B_chosen_cj / B_nb)
        mean_low_cv.append(chosen_value_low / low_cv)
        mean_medium_cv.append(chosen_value_medium / medium_cv)
        mean_high_cv.append(chosen_value_high / high_cv)
        mean_i_stim_choice_a.append(i_stim_choice_a / A_nb)
        mean_i_stim_choice_b.append(i_stim_choice_b / B_nb)
        mean_i_ampa_ext_choice_a.append(i_ampa_ext_choice_a / A_nb)
        mean_i_ampa_ext_choice_b.append(i_ampa_ext_choice_b / B_nb)
        mean_i_ampa_ext_a_choice_a.append(i_ampa_ext_a_choice_a / A_nb)
        mean_i_ampa_ext_a_choice_b.append(i_ampa_ext_a_choice_b / B_nb)
    return (ovb_rate_low, ovb_rate_medium, ovb_rate_high, mean_A_chosen_cj, mean_B_chosen_cj, mean_low_cv, mean_medium_cv, mean_high_cv,
            mean_i_stim_choice_a, mean_i_stim_choice_b, mean_i_ampa_ext_choice_a, mean_i_ampa_ext_choice_b, mean_i_ampa_ext_a_choice_a, mean_i_ampa_ext_a_choice_b)



def graph():
    (ovb_rate_low, ovb_rate_medium, ovb_rate_high, mean_A_chosen_cj, mean_B_chosen_cj, mean_low_cv, mean_medium_cv,
     mean_high_cv, mean_i_stim_choice_a, mean_i_stim_choice_b, mean_i_ampa_ext_choice_a, mean_i_ampa_ext_choice_b, mean_i_ampa_ext_a_choice_a, mean_i_ampa_ext_a_choice_b) = result_firing_rate()
    X_axis = np.arange(0, 1.5, 0.0005)
    bokeh.plotting.output_notebook()
    figure_4_A = bokeh.plotting.figure(title="Figure 4 A", plot_width=300, plot_height=300)
    figure_4_E = bokeh.plotting.figure(title="Figure 4 E", plot_width=300, plot_height=300)
    figure_4_I = bokeh.plotting.figure(title="Figure 4 I", plot_width=300, plot_height=300)
    figure_i_stim =  bokeh.plotting.figure(title="Figure I stim CJ B", plot_width=300, plot_height=300)
    figure_i_ampa_ext = bokeh.plotting.figure(title="Figure I ampa ext CJ B", plot_width=300, plot_height=300)
    figure_i_ampa_ext_a = bokeh.plotting.figure(title="Figure I ampa ext CJ A", plot_width=300, plot_height=300)

    figure_4_A.multi_line([X_axis, X_axis, X_axis], [ovb_rate_low, ovb_rate_medium, ovb_rate_high] , color =['red', "green", "blue"])
    figure_4_E.multi_line([X_axis, X_axis] , [mean_A_chosen_cj, mean_B_chosen_cj], color =['red', "blue"])
    figure_4_I.multi_line([X_axis, X_axis, X_axis] , [mean_low_cv, mean_medium_cv, mean_high_cv], color =['red', "green", "blue"])
    figure_i_stim.multi_line([X_axis, X_axis] , [mean_i_stim_choice_a, mean_i_stim_choice_b], color =['red', "blue"])
    figure_i_ampa_ext.multi_line([X_axis, X_axis], [mean_i_ampa_ext_choice_a, mean_i_ampa_ext_choice_b], color=['red', "blue"])
    figure_i_ampa_ext_a.multi_line([X_axis, X_axis], [mean_i_ampa_ext_a_choice_a, mean_i_ampa_ext_a_choice_b], color=['red', "blue"])

    bokeh.plotting.show(figure_4_A)
    bokeh.plotting.show(figure_4_E)
    bokeh.plotting.show(figure_4_I)
    bokeh.plotting.show(figure_i_stim)
    bokeh.plotting.show(figure_i_ampa_ext)
    bokeh.plotting.show(figure_i_ampa_ext_a)


graph()



#ovb_rate_low, ovb_rate_medium, ovb_rate_high = result_firing_rate()


#(r_ov_a , r_ov_b, r_i_cj_a, S_cj_a, r_i_cj_b, S_cj_b, r_i_ns, S_ns, r_i_cv_cells, S_gaba_cv) = one_trial(0, 20, [0, 0], [20, 20], 0, 0, 0, 0, 0,
#                0,0,0,0, [0, 0, 0], [ 0, 0, 0], [0, 0, 0], 0)
#X_axis = range(len(mean_ov_b))




#s_ampa_cj_b_tot, s_nmda_b_tot, s_gaba_b_tot, s_gaba_cv_tot)
#bokeh.plotting.output_notebook()
#figure_4_A = bokeh.plotting.figure(title="Figure 4 A", plot_width=300, plot_height=300)
#figure_4_E = bokeh.plotting.figure(title="Figure 4 E", plot_width=300, plot_height=300)
#figure_4_I = bokeh.plotting.figure(title="Figure 4 I", plot_width=300, plot_height=300)
#figure_4_ns = bokeh.plotting.figure(title="Figure 4 ns", plot_width=300, plot_height=300)
#figure_4_A.line(X_axis , mean_ov_b, color ='red')
#figure_4_E.line(X_axis , r_i_cj_a_tot, color ='red')
#figure_4_I.line(X_axis , r_i_cj_b_tot, color ='red')
#figure_4_ns.line(X_axis, r_i_ns_tot, color = 'red')
#graph_ampa = bokeh.plotting.figure(title="ampa b", plot_width=300, plot_height=300)
#graph_nmda = bokeh.plotting.figure(title="nmda b", plot_width=300, plot_height=300)
#graph_gaba = bokeh.plotting.figure(title="gaba b", plot_width=300, plot_height=300)
#graph_gaba_cv = bokeh.plotting.figure(title="gaba cv b", plot_width=300, plot_height=300)
#graph_ampa.line(X_axis , s_ampa_cj_b_tot, color ='blue')
#graph_nmda.line(X_axis , s_nmda_b_tot, color ='blue')
#graph_gaba.line(X_axis , s_gaba_b_tot, color ='blue')
#graph_gaba_cv.line(X_axis , s_gaba_cv_tot, color ='blue')

#bokeh.plotting.show(figure_4_A)
#bokeh.plotting.show(figure_4_E)
#bokeh.plotting.show(figure_4_I)
#bokeh.plotting.show(figure_4_ns)
#bokeh.plotting.show(graph_ampa)
#bokeh.plotting.show(graph_nmda)
#bokeh.plotting.show(graph_gaba)
#bokeh.plotting.show(graph_gaba_cv)



    ## Notes

"""
* In equations 11, 12, 13, 15, 16 and 17, _pyr and _in suffixes have been corrected to _rec_in and
_pyr_in respectively.
* In equation 17, I_gaba_rec_i has been corrected into I_gaba_rec_I.
* In the parameters, F has been corrected into f
"""
