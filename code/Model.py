#-*- coding: utf-8 -*-

import numpy as np
import math
import collections # to order items in dictionary result
import bokeh
import bokeh.plotting


def firing_pyr_cells(r_i, phi, τ_ampa, dt): #1
    """Update the firing rate of pyramidale cells (eq. 1)"""
    r_i += ((- r_i + phi) / τ_ampa ) * dt
    return r_i

def firing_rate_I(r_I, phi, τ_gaba, dt): #2
    """Update the firing rate of interneurons (eq. 2)"""
    r_I += ((-r_I + phi) / τ_gaba) * dt
    return r_I

def channel_ampa(S_ampa, τ_ampa, r_i, dt): #3
    """Open AMPA channels (eq. 3)"""
    S_ampa += ((- S_ampa / τ_ampa) + r_i) * dt
    return S_ampa

def channel_nmda(S_nmda, S_nmda_1, τ_nmda, γ, r_i, dt): #4
    """Open NMDA channels (eq. 4)"""
    S_nmda += ((-S_nmda / τ_nmda) + (1 - S_nmda_1) * γ * r_i) * dt
    return S_nmda

def channel_gaba(S_gaba, τ_gaba, r_I, dt): #5
    """Open GABA channels (eq. 5)"""
    S_gaba += (-S_gaba / τ_gaba + r_I) * dt
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

def I_gaba_rec_I(N_I, J_gaba_in, S_gaba): #17
    """Compute the recurrent GABA current for interneurons (eq. 17)"""
    return -N_I * J_gaba_rec_in * S_gaba

def eta(t):
    """Ornstein-Uhlenbeck process (here just Gaussian random noise)"""
    return np.random.normal(0, 1)
    return normale

def white_noise(I_eta, τ_ampa, σ_eta, t, dt): #18
    """Update I_eta, the noise term (eq. 18)"""
    I_eta += ((-I_eta + eta(t) * math.sqrt(τ_ampa * (σ_eta ** 2))) / τ_ampa) * dt
    return I_eta

def I_stim(J_ampa_input, δ_j_hl, δ_j_stim, τ_ampa, r_ov): #19
    """Computng the primary input (eq. 19)"""
    return -J_ampa_input * δ_j_hl * δ_j_stim * τ_ampa * r_ov

def firing_ov_cells(x, xmin, x_max, t, b, d, r_o, Δ_r): #20, 21, 22, 23
    """Computing the activity profile of OV cells (eq. 20, 21, 22, 23)"""
    x_i = (x - xmin) / (x_max - xmin)
    g_t = (1 / (1 + np.exp(- (t - a) / b))) * (1 / (1 + np.exp((t - c) / d)))
    f_t = g_t / g_max
    r_ov = r_o + Δ_r * f_t * x_i
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
t_offer = 0 # s
a = t_offer + 0.175 # s
b = 0.030 # s
c = t_offer + 0.400 # s
d = 0.100 # s
J_ampa_input = 30 * J_ampa_ext_pyr

dt = 0.0005 # s


    ##

g_max = 0
list_g = []
T = np.arange(0, 1, dt)
for T_i in T:
    g = (1 / (1 + np.exp(-(T_i - a) / b))) * (1 / (1 + np.exp((T_i - c) / d)))
    list_g.append(g)
g_max += np.max(list_g)



w_p = 1.75
w_m = 1 - f * (w_p - 1) / (1 - f)




δ_j_hl_cj_a, δ_j_hl_cj_b = 1, 1
δ_j_stim_cj_a, δ_j_stim_cj_b = 2, 1
δ_J_gaba_cj_a, δ_J_gaba_cj_b, δ_J_gaba_ns = 1, 1, 1
δ_J_nmda_cj_a, δ_J_nmda_cj_b = 1, 1

I_stim_cv, I_stim_ns = 0, 0


def ov_a_cells(g_list_a, t, x_a, xmin, x_max):

    r_ov_a = firing_ov_cells(x_a, xmin, x_max, t, b, d, r_o, Δ_r)
    return r_ov_a


def ov_b_cells(g_list_b, t, x_b, xmin, x_max):

    r_ov_b = firing_ov_cells(x_b, xmin, x_max, t, b, d, r_o, Δ_r)
    return r_ov_b


def cj_a_cells(t, r_i_cj_a, S_ampa_cj_a, S_nmda_cj_a, S_gaba_cj_a, I_eta_cj_a,
               S_ampa_cj_b, S_ampa_ns, S_nmda_cj_b, S_nmda_ns, r_i_cv_cells, r_ov_a):

    S_ampa_cj_a = channel_ampa(S_ampa_cj_a, τ_ampa, r_i_cj_a, dt)
    S_nmda_cj_a = channel_nmda(S_nmda_cj_a, S_nmda_cj_a, τ_nmda, γ, r_i_cj_a, dt)
    S_gaba_cj_a = channel_gaba(S_gaba_cj_a, τ_gaba, r_i_cv_cells, dt)
    S_cj_a = [S_ampa_cj_a, S_nmda_cj_a, S_gaba_cj_a]

    I_eta_cj_a = white_noise(I_eta_cj_a, τ_ampa, σ_eta, t, dt)
    I_ampa_ext_cj_a = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_cj_a)
    I_ampa_rec_cj_a = I_ampa_rec(N_E, f, J_ampa_rec_pyr, w_p, w_m, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns)
    I_nmda_rec_cj_a = I_nmda_rec(N_E, f, J_nmda_rec_pyr, δ_J_nmda_cj_a, w_p, S_nmda_cj_a, w_m, S_nmda_cj_b, S_nmda_ns)
    I_gaba_rec_cj_a = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_cj_a, S_gaba_cj_a)
    I_stim_cj_a = I_stim(J_ampa_input, δ_j_hl_cj_a, δ_j_stim_cj_a, τ_ampa, r_ov_a)
    I_syn_cj_a = I_syn(I_ampa_ext_cj_a, I_ampa_rec_cj_a, I_nmda_rec_cj_a, I_gaba_rec_cj_a, I_stim_cj_a)

    phi_cj_a = Φ(I_syn_cj_a, c_E, g_E, I_E)

    r_i_cj_a = firing_pyr_cells(r_i_cj_a, phi_cj_a, τ_ampa, dt)
    return r_i_cj_a, S_cj_a, I_eta_cj_a


def cj_b_cells(t, r_i_cj_b, S_ampa_cj_b, S_nmda_cj_b, S_gaba_cj_b, I_eta_cj_b,
               S_ampa_cj_a, S_ampa_ns, S_nmda_cj_a, S_nmda_ns, r_i_cv_cells, r_ov_b):

    S_ampa_cj_b = channel_ampa(S_ampa_cj_b, τ_ampa, r_i_cj_b, dt) # equation 3
    S_nmda_cj_b = channel_nmda(S_nmda_cj_b, S_nmda_cj_a, τ_nmda, γ, r_i_cj_b, dt) # equation 4
    S_gaba_cj_b = channel_gaba(S_gaba_cj_b, τ_gaba, r_i_cv_cells, dt) # equation 5
    S_cj_b = [S_ampa_cj_b, S_nmda_cj_b, S_gaba_cj_b]
    print('S_ampa_cj_b, S_nmda_cj_b, S_gaba_cj_b', S_cj_b)
    I_eta_cj_b = white_noise(I_eta_cj_b, τ_ampa, σ_eta, t, dt) # equation 18
    I_ampa_ext_cj_b = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_cj_b) # equation 8
    I_ampa_rec_cj_b = I_ampa_rec(N_E, f, J_ampa_rec_pyr, w_p, w_m, S_ampa_cj_b, S_ampa_cj_a, S_ampa_ns) # equation 9
    I_nmda_rec_cj_b = I_nmda_rec(N_E, f, J_nmda_rec_pyr, δ_J_nmda_cj_b, w_p, S_nmda_cj_a, w_m, S_nmda_cj_b, S_nmda_ns) # equation 11
    I_gaba_rec_cj_b = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_cj_b, S_gaba_cj_b) # equation 13
    I_stim_cj_b = I_stim(J_ampa_input, δ_j_hl_cj_b, δ_j_stim_cj_b, τ_ampa, r_ov_b) # equation 19

    I_syn_cj_b = I_syn(I_ampa_ext_cj_b, I_ampa_rec_cj_b, I_nmda_rec_cj_b, I_gaba_rec_cj_b, I_stim_cj_b) #equation 7

    print('I_syn_cj_b', I_syn_cj_b)
    print( 'I_ampa_ext_cj_b, I_ampa_rec_cj_b, I_nmda_rec_cj_b, I_gaba_rec_cj_b, I_stim_cj_b',
           I_ampa_ext_cj_b, I_ampa_rec_cj_b, I_nmda_rec_cj_b, I_gaba_rec_cj_b, I_stim_cj_b)

    phi_cj_b = Φ(I_syn_cj_b, c_E, g_E, I_E) #equation 6
    print('phi_cj_b', phi_cj_b)
    r_i_cj_b = firing_pyr_cells(r_i_cj_b, phi_cj_b, τ_ampa, dt) #equation 1
    return r_i_cj_b, S_cj_b, I_eta_cj_b


def ns_cells(t, r_i_ns, S_ampa_ns, S_nmda_ns, S_gaba_ns, I_eta_ns,
             S_ampa_cj_a, S_ampa_cj_b, S_nmda_cj_a, S_nmda_cj_b, r_i_cv_cells):

    S_ampa_ns = channel_ampa(S_ampa_ns, τ_ampa, r_i_ns, dt) #equation 3
    S_nmda_ns = channel_nmda(S_nmda_ns, S_nmda_cj_a, τ_nmda, γ, r_i_ns, dt) #equation 4
    S_gaba_ns = channel_gaba(S_gaba_ns, τ_gaba, r_i_cv_cells, dt) #equation 5
    S_ns = [S_ampa_ns, S_nmda_ns, S_gaba_ns]

    I_eta_ns = white_noise(I_eta_ns, τ_ampa, σ_eta, t, dt) #equation 18
    I_ampa_ext_ns = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_ns) #equation 8
    I_ampa_rec_ns = I_ampa_rec_3(N_E, f, J_ampa_rec_pyr, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns) #equation 10
    I_nmda_rec_ns = I_nmda_rec_3(N_E, f, J_nmda_rec_pyr, S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns) #equation 12
    I_gaba_rec_ns = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_ns, S_gaba_ns) #equation 13

    I_syn_ns = I_syn(I_ampa_ext_ns, I_ampa_rec_ns, I_nmda_rec_ns, I_gaba_rec_ns, I_stim_ns) #equation 7
    phi_ns = Φ(I_syn_ns, c_E, g_E, I_E) #equation 6
    r_i_ns = firing_pyr_cells(r_i_ns, phi_ns, τ_ampa, dt) #equation 1
    return r_i_ns, S_ns, I_eta_ns



def cv_cells(t, r_i_cv_cells, S_gaba_cv, I_eta_cv,
             S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns,
             S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns):

    S_gaba_cv = channel_gaba(S_gaba_cv, τ_gaba, r_i_cv_cells, dt) #equation 5

    I_eta_cv = white_noise(I_eta_cv, τ_ampa, σ_eta, t, dt)  #equation 18
    I_ampa_ext_cv = I_ampa_ext_I(J_ampa_ext_in, τ_ampa, C_ext, r_ext, I_eta_cv)  #equation 14
    I_ampa_rec_cv = I_ampa_rec_I(N_E, f, J_ampa_rec_in, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns) #equation 15
    I_nmda_rec_cv = I_nmda_rec_I(N_E, f, J_nmda_rec_in, S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns) #equation 16
    I_gaba_rec_cv = I_gaba_rec_I(N_I, J_gaba_rec_in, S_gaba_cv) #equation 17
    I_syn_cv_cells = I_syn(I_ampa_ext_cv, I_ampa_rec_cv, I_nmda_rec_cv, I_gaba_rec_cv, I_stim_cv) #equation 7

    phi_cv_cells = Φ(I_syn_cv_cells, c_I, g_I, I_I) #equation 6
    r_i_cv_cells += firing_rate_I(r_i_cv_cells, phi_cv_cells, τ_gaba, dt) #equation 2
    return r_i_cv_cells, S_gaba_cv, I_eta_cv


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


def one_trial(g_list_a, g_list_b, x_a, x_b, xmin_list, x_max_list, t,
              r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
              I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
              S_cj_a, S_cj_b, S_ns, S_gaba_cv):

    mean_ov_b = []
    while t < 0.010:

        r_ov_a = ov_a_cells(g_list_a, t, x_a, xmin_list[0], x_max_list[0])
        r_ov_b = ov_b_cells(g_list_b, t, x_b, xmin_list[1], x_max_list[1])

        r_i_cj_a, S_cj_a, I_eta_cj_a = cj_a_cells(t, r_i_cj_a, S_cj_a[0], S_cj_a[1], S_cj_a[2],
                                                 I_eta_cj_a, S_cj_b[0], S_ns[0], S_cj_b[1], S_ns[1], r_i_cv_cells, r_ov_a)
        r_i_cj_b, S_cj_b, I_eta_cj_b = cj_b_cells(t, r_i_cj_b, S_cj_b[0], S_cj_b[1], S_cj_b[2],
                                                 I_eta_cj_b, S_cj_a[0], S_ns[0], S_cj_a[1], S_ns[1], r_i_cv_cells, r_ov_b)
        r_i_ns, S_ns, I_eta_ns = ns_cells(t, r_i_ns, S_ns[0], S_ns[1], S_ns[2],
                                         I_eta_ns,S_cj_a[0], S_cj_b[0], S_cj_a[1], S_cj_b[1], r_i_cv_cells)
        r_i_cv_cells, S_gaba_cv, I_eta_cv = cv_cells(t, r_i_cv_cells, S_gaba_cv,
                                                    I_eta_cv, S_cj_a[0], S_cj_b[0], S_ns[0], S_cj_a[1], S_cj_b[1], S_ns[1])
        mean_ov_b.append(r_ov_b)
        print('r_ov_b', r_ov_b)
        print('x_a, x_b, r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells', x_a, x_b, r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells)
        t += dt

    if r_i_cj_a > r_i_cj_b :
        choice = 'choice A'
    elif r_i_cj_a < r_i_cj_b:
        choice = 'choice B'
    else :
        return 'error in choice'
    print(r_i_cj_a)
    return choice, mean_ov_b


def session():
    r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells = 0, 0, 0, 0
    I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv = 0, 0, 0, 0
    S_cj_a, S_cj_b, S_ns = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    S_gaba_cv = 0
    t = 0.0005
    quantity_a, quantity_b = [], []
    g_list_a, g_list_b = [], []
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
    for i in range(1):
        choice, mean_ov_b = one_trial(g_list_a, g_list_b, quantity_a[i], quantity_b[i], xmin_list, x_max_list, t,
                                      r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
                                      I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
                                      S_cj_a, S_cj_b, S_ns, S_gaba_cv)
        result[(x_a, x_b)].append([choice, mean_ov_b])
        print(i)
    return result


print(session())

def result_firing_rate():

    ''' on obtient la moyenne des ov_b rate en fonction du temps
    et si l'essai a eu une offre forte, moyenne, faible '''

    ovb_rate_low, ovb_rate_high, ovb_rate_medium = [], [], []
    low, medium, high = 0, 0, 0
    result = session()

    ''' le terme k représente le temps,
     le terme i représente la quantité de A,
      le j représente la quantité de B
      et le l représente la liste de l'essai l
      pour un temps donné, on ajoute les r_ov_b pour chaque (i,j) pour chaque essai l '''

    for k in range(2000):
        mean_low, mean_high, mean_medium = 0, 0, 0
        for i in range(20):
            for j in range(6):
                    for l in range(len(result[(i,j)])):
                        mean_low += result[(i,j)][l][1][k]
                        low += 1
            for j in range(7,13):
                    for l in range(len(result[(i,j)])):
                        mean_medium += result[(i,j)][l][1][k]
                        medium += 1
            for j in range(14,20):
                    for l in range(len(result[(i,j)])):
                        mean_high += result[(i,j)][l][1][k]
                        high += 1
        ovb_rate_low.append(mean_low / low)
        ovb_rate_medium.append(mean_medium / medium)
        ovb_rate_high.append(mean_high / high)
    return ovb_rate_low, ovb_rate_medium, ovb_rate_high

#ovb_rate_low, ovb_rate_medium, ovb_rate_high = result_firing_rate()
#X_axis = range(2000)


#bokeh.plotting.output_notebook()
#figure_4 = bokeh.plotting.figure(title="Figure 4 A", plot_width=300, plot_height=300)
#figure_4.multi_line([X_axis, X_axis, X_axis] , [ovb_rate_low, ovb_rate_medium, ovb_rate_high], color =['blue', 'green', 'red'] )


    ## Notes

"""
* In equations 11, 12, 13, 15, 16 and 17, _pyr and _in suffixes have been corrected to _rec_in and
_pyr_in respectively.
* In equation 17, I_gaba_rec_i has been corrected into I_gaba_rec_I.
* In the parameters, F has been corrected into f
"""
