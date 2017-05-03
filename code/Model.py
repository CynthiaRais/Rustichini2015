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
    return r_i

def firing_rate_I(r_I, phi, τ_gaba, dt): #2
    """Update the firing rate of interneurons (eq. 2)"""
    r_I += ((-r_I + phi) / τ_gaba) * dt
    return r_I

def channel_ampa(S_ampa, τ_ampa, r_i, dt): #3
    """Open AMPA channels (eq. 3)"""
    S_ampa += ((- S_ampa / τ_ampa) + r_i) * dt
    return S_ampa

def channel_nmda(S_nmda, τ_nmda, γ, r_i, dt): #4
    """Open NMDA channels (eq. 4)"""
    S_nmda += ((-S_nmda / τ_nmda) + (1 - S_nmda) * γ * r_i) * dt
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

    f_t = g_t / g_max
    r_ov = r_o + Δ_r * f_t * x_i
    return r_ov


def logistic_model(a_0, a_1, a_2, a_3, a_4, a_5, quantity_a, quantity_b ): #25
    """Computing the logistic model of figure 4 to examine departures from linearity"""
    X = a_0 + a_1 * quantity_a + a_2 * quantity_b + a_3 * (quantity_a) ** 2 + a_4 * (quantity_b) ** 2 + a_5 * (quantity_a * quantity_b)
    choice_B = 1 / (1 + np.exp(-X))
    return choice_B



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
t_offer = 1.0 # s
a = t_offer + 0.175 # s
b = 0.030 # s
c = t_offer + 0.400 # s
d = 0.100 # s
J_ampa_input = 30 * J_ampa_ext_pyr

dt = 0.0005 # s


    ##


list_g = []
for t in np.arange(0, 2.0, dt):
    g = (1 / (1 + np.exp(- (t- a) / b))) * (1 / (1 + np.exp((t - c) / d)))
    list_g.append(g)
g_max = np.max(list_g)



w_p = 1.75
w_m = 1 - f * (w_p - 1) / (1 - f)




δ_j_hl_cj_a, δ_j_hl_cj_b = 1, 1
δ_j_stim_cj_a, δ_j_stim_cj_b = 2, 1
δ_J_gaba_cj_a, δ_J_gaba_cj_b, δ_J_gaba_ns = 1, 1, 1
δ_J_nmda_cj_a, δ_J_nmda_cj_b = 1, 1




def cj_cells(r_i_cj, S_ampa_cj, S_nmda_cj, S_gaba_cj, I_eta_cj,
               S_ampa_cj_2, S_ampa_ns, S_nmda_cj_2, S_nmda_ns, r_i_cv_cells, r_ov):
    """Compute firing rate of CJA and CJB cells"""

    S_ampa_cj = channel_ampa(S_ampa_cj, τ_ampa, r_i_cj, dt) #equation 3
    S_nmda_cj = channel_nmda(S_nmda_cj, τ_nmda, γ, r_i_cj, dt) #equation 4
    S_gaba_cj = channel_gaba(S_gaba_cj, τ_gaba, r_i_cv_cells, dt) #equation 5
    S_cj = [S_ampa_cj, S_nmda_cj, S_gaba_cj]

    I_eta_cj = white_noise(I_eta_cj, τ_ampa, σ_eta, dt) #equation 18
    I_ampa_ext_cj = I_ampa_ext(J_ampa_ext_pyr, τ_ampa, C_ext, r_ext, I_eta_cj) #equation 8
    I_ampa_rec_cj = I_ampa_rec(N_E, f, J_ampa_rec_pyr, w_p, w_m, S_ampa_cj, S_ampa_cj_2, S_ampa_ns) #equation 9
    I_nmda_rec_cj = I_nmda_rec(N_E, f, J_nmda_rec_pyr, δ_J_nmda_cj_a, w_p, S_nmda_cj, w_m, S_nmda_cj_2, S_nmda_ns) #equation 11
    I_gaba_rec_cj = I_gaba_rec(N_I, J_gaba_rec_pyr, δ_J_gaba_cj_a, S_gaba_cj) #equation 13
    I_stim_cj = I_stim(J_ampa_input, δ_j_hl_cj_a, δ_j_stim_cj_a, τ_ampa, r_ov) #equation 19
    I_syn_cj = I_syn(I_ampa_ext_cj, I_ampa_rec_cj, I_nmda_rec_cj, I_gaba_rec_cj, I_stim_cj) #equation 7

    phi_cj = Φ(I_syn_cj, c_E, g_E, I_E) #equation 6
    r_i_cj = firing_pyr_cells(r_i_cj, phi_cj, τ_gaba, dt) #equation 1
    return r_i_cj, S_cj


def ns_cells(r_i_ns, S_ampa_ns, S_nmda_ns, S_gaba_ns, I_eta_ns,
             S_ampa_cj_a, S_ampa_cj_b, S_nmda_cj_a, S_nmda_cj_b, r_i_cv_cells):
    """Compute firing rate of NS cells"""

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
    return r_i_ns, S_ns



def cv_cells(r_i_cv_cells, S_gaba_cv, I_eta_cv,
             S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns,
             S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns):
    """Compute firing rate of CV cells"""

    S_gaba_cv = channel_gaba(S_gaba_cv, τ_gaba, r_i_cv_cells, dt) #equation 5

    I_eta_cv = white_noise(I_eta_cv, τ_ampa, σ_eta, dt)  #equation 18
    I_ampa_ext_cv = I_ampa_ext_I(J_ampa_ext_in, τ_ampa, C_ext, r_ext, I_eta_cv)  #equation 14
    I_ampa_rec_cv = I_ampa_rec_I(N_E, f, J_ampa_rec_in, S_ampa_cj_a, S_ampa_cj_b, S_ampa_ns) #equation 15
    I_nmda_rec_cv = I_nmda_rec_I(N_E, f, J_nmda_rec_in, S_nmda_cj_a, S_nmda_cj_b, S_nmda_ns) #equation 16
    I_gaba_rec_cv = I_gaba_rec_I(N_I, J_gaba_rec_in, S_gaba_cv) #equation 17
    I_stim_cv = 0
    I_syn_cv_cells = I_syn(I_ampa_ext_cv, I_ampa_rec_cv, I_nmda_rec_cv, I_gaba_rec_cv, I_stim_cv) #equation 7

    phi_cv_cells = Φ(I_syn_cv_cells, c_I, g_I, I_I) #equation 6
    r_i_cv_cells = firing_rate_I(r_i_cv_cells, phi_cv_cells, τ_gaba, dt) #equation 2
    return r_i_cv_cells, S_gaba_cv


def quantity_juice():
    """random choice of juice quantity, ΔA = ΔB = [0, 20] """

    x_a = np.random.randint(0, 21)
    x_b = np.random.randint(0, 21)
    while x_a == 0 and x_b == 0:
        p = np.random.random()
        if p < 0.5:
            x_a = np.random.randint(0, 21)
        else:
            x_b = np.random.randint(0, 21)
    return x_a, x_b


def one_trial(x_a, x_b, xmin_list, x_max_list,
              r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
              I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
              S_cj_a, S_cj_b, S_ns, S_gaba_cv):
    """Compute one trial"""

    ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial = [], [], []

    for t in np.arange(0, 2.0, 0.0005):
        """Firing rate of OV cells"""
        r_ov_a = firing_ov_cells(x_a, xmin_list[0], x_max_list[0], t, r_o, Δ_r)
        r_ov_b = firing_ov_cells(x_b, xmin_list[1], x_max_list[1], t, r_o, Δ_r)

        """Firing rate of CJA and CJB cells"""
        r_i_cj_a, S_cj_a = cj_cells(r_i_cj_a, S_cj_a[0], S_cj_a[1], S_cj_a[2],
                                    I_eta_cj_a, S_cj_b[0], S_ns[0], S_cj_b[1], S_ns[1], r_i_cv_cells, r_ov_a)
        r_i_cj_b, S_cj_b = cj_cells(r_i_cj_b, S_cj_b[0], S_cj_b[1], S_cj_b[2],
                                    I_eta_cj_b, S_cj_a[0], S_ns[0], S_cj_a[1], S_ns[1], r_i_cv_cells, r_ov_b)

        """Firing rate of NS cells"""
        r_i_ns, S_ns = ns_cells(r_i_ns, S_ns[0], S_ns[1], S_ns[2],
                                I_eta_ns,S_cj_a[0], S_cj_b[0], S_cj_a[1], S_cj_b[1], r_i_cv_cells)

        """Firing rate of CV cells"""
        r_i_cv_cells, S_gaba_cv = cv_cells(r_i_cv_cells, S_gaba_cv,
                                            I_eta_cv, S_cj_a[0], S_cj_b[0], S_ns[0], S_cj_a[1], S_cj_b[1], S_ns[1])

        ov_b_one_trial.append(r_ov_b)
        r_i_cj_b_one_trial.append(r_i_cj_b)
        r_i_cv_cells_one_trial.append(r_i_cv_cells)

    """Determine the final choice"""
    if r_i_cj_a > r_i_cj_b :
        choice = 'choice A'
    elif r_i_cj_a < r_i_cj_b:
        choice = 'choice B'
    else :
        raise ValueError(choice ='no choice')

    return choice, ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial


def session():
    n = 100  #number of trials for a session
    """Create and order the dictionary result"""
    result = {}
    for j in range(0, 21):
        for k in range(0, 21):
                result[(j, k)] = []
    result = collections.OrderedDict(sorted(result.items(), key = lambda t: t[0]))

    """Determine the quantity of each juice for all the trials in the session"""
    quantity_a, quantity_b = [], []
    for i in range(n):
        x_a, x_b = quantity_juice()
        quantity_a.append(x_a)
        quantity_b.append(x_b)

    """Determine the min and max quantity for each juice in the session"""
    xmin_a = np.min(quantity_a)
    xmin_b = np.min(quantity_b)
    x_max_a = np.max(quantity_a)
    x_max_b = np.max(quantity_b)
    xmin_list = [xmin_a, xmin_b]
    x_max_list = [x_max_a, x_max_b]

    for i in range(n):
        '''for graphs until figure 7, all parameters are reset to 0 at the beginning of each trial'''
        r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells = 0, 0, 0, 0
        I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv = 0, 0, 0, 0
        S_cj_a, S_cj_b, S_ns = [0, 0, 0], [0, 0, 0], [0, 0, 0]
        S_gaba_cv = 0
        choice, ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial = one_trial(quantity_a[i], quantity_b[i], xmin_list, x_max_list,
                                                        r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
                                                        I_eta_cj_a, I_eta_cj_b, I_eta_ns, I_eta_cv,
                                                        S_cj_a, S_cj_b, S_ns, S_gaba_cv)

        result[(quantity_a[i], quantity_b[i])].append([choice, ov_b_one_trial, r_i_cj_b_one_trial, r_i_cv_cells_one_trial])
    return result



def result_firing_rate():

    """ on obtient la moyenne des ov_b rate en fonction du temps
    et si l'essai a eu une offre forte, moyenne, faible """

    ovb_rate_low, ovb_rate_high, ovb_rate_medium = [], [], []

    result = session()
    mean_A_chosen_cj, mean_B_chosen_cj = [], []
    mean_low_cv, mean_medium_cv, mean_high_cv = [], [], []
    ''' le terme k représente le temps,
     le terme i représente la quantité de A,
      le j représente la quantité de B
      et le l représente la liste de l'essai l
      pour un temps donné, on ajoute les r_ov_b pour chaque (i,j) pour chaque essai l
      (figure 4A)'''

    for k in range(4000):
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

    '''mean depending on choice (figure 4E, 4I)'''
    for k in range (4000):
        A_chosen_cj, B_chosen_cj = 0, 0
        chosen_value_low, chosen_value_medium, chosen_value_high = 0, 0, 0
        A_nb, B_nb = 0, 0
        low_cv, medium_cv, high_cv = 0, 0, 0
        for i in range(21):
            for j in range(21):
                for l in range(len(result[(i,j)])):
                    if result[(i, j)][l][0] == 'choice A':
                        A_chosen_cj += result[(i, j)][l][2][k]
                        A_nb +=1
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
    return (ovb_rate_low, ovb_rate_medium, ovb_rate_high, mean_A_chosen_cj, mean_B_chosen_cj, mean_low_cv, mean_medium_cv,
    mean_high_cv)

    #""" pour les figures C, G, K"""
    #mean_ov_fig_C_A_ji, mean_ov_fig_C_A_ij, mean_ov_fig_C_B_ji, mean_ov_fig_C_B_ij =[], [], [], []
    #mean_cj_fig_G_A_ji, mean_cj_fig_G_A_ij, mean_cj_fig_G_B_ji, mean_cj_fig_G_B_ij = [], [], [], []
    #mean_cv_fig_K_A_ji, mean_cv_fig_K_A_ij, mean_cv_fig_K_B_ji, mean_cv_fig_K_B_ij = [], [], [], []
    #'''focus on the time window after the offer : 0 - 500 ms for OV and CJ cells and on 500 ms - 1000 ms for CV cells'''

#    for i in range(1,2):
#        for j in range(20, 3, -2):
#            print("j=", j)
#            ov_fig_C_A_ji, ov_fig_C_B_ji, cj_fig_G_A_ji, cj_fig_G_B_ji = 0, 0, 0, 0
#            cv_fig_K_A_ji, cv_fig_K_B_ji = 0, 0
#            ov_fig_C_A_ij, ov_fig_C_B_ij, cj_fig_G_A_ij, cj_fig_G_B_ij = 0, 0, 0, 0
#            cv_fig_K_A_ij, cv_fig_K_B_ij = 0, 0
#            A_number, A_number_ij, B_number, B_number_ij = 0, 0, 0, 0
#            for k in range(2000, 3000):
#                for l in range(len(result[(j,i)])):
#                    if result[(j, i)][l][0] == 'choice A':
#                        ov_fig_C_A_ji += result[(j, i)][l][1][k]
#                        cj_fig_G_A_ji += result[(j, i)][l][2][k]
#                        cv_fig_K_A_ji += result[(j, i)][l][3][k + 1000]
#                        A_number +=1

#                    else :
#                        ov_fig_C_B_ji += result[(j, i)][l][1][k]
#                        cj_fig_G_B_ji += result[(j, i)][l][2][k]
#                        cv_fig_K_B_ji += result[(j, i)][l][3][k + 1000]
#                        B_number +=1

#                for l in range(len(result[(i,j)])):
#                    if result[(i,j)][l][0] == 'choice A':
#                        ov_fig_C_A_ij += result[(i, j)][l][1][k]
#                        cj_fig_G_A_ij += result[(i, j)][l][2][k]
#                        cv_fig_K_A_ij += result[(i, j)][l][2][k + 1000]
#                        A_number_ij +=1
#                    else :
#                        ov_fig_C_B_ij += result[(i, j)][l][1][k]
#                        cj_fig_G_B_ij += result[(i, j)][l][2][k]
#                        cv_fig_K_B_ij += result[(i, j)][l][2][k + 1000]
#                        B_number_ij += 1
#            mean_ov_fig_C_A_ji.append(ov_fig_C_A_ji / A_number)
#            mean_ov_fig_C_B_ji.append(ov_fig_C_B_ji / B_number)
#            mean_cj_fig_G_A_ji.append(cj_fig_G_A_ji / A_number)
#            mean_cj_fig_G_B_ji.append(cj_fig_G_A_ji / B_number)
#            mean_cv_fig_K_A_ji.append(cv_fig_K_A_ji / A_number)
#            mean_cv_fig_K_B_ji.append(cv_fig_K_B_ji / B_number)
#            mean_ov_fig_C_A_ij.append(ov_fig_C_A_ij / A_number_ij)
#            mean_ov_fig_C_B_ij.append(ov_fig_C_B_ij / B_number_ij)
#            mean_cj_fig_G_A_ij.append(cj_fig_G_A_ij / A_number_ij)
#            mean_cj_fig_G_B_ij.append(cj_fig_G_B_ij / B_number_ij)
#            mean_cv_fig_K_A_ij.append(cv_fig_K_A_ij / A_number_ij)
#            mean_cv_fig_K_B_ij.append(cv_fig_K_B_ij / B_number_ij)
#    mean_ov_fig_C_A = mean_ov_fig_C_A_ji + mean_ov_fig_C_A_ij[::-1]
#    mean_ov_fig_C_B = mean_ov_fig_C_B_ji + mean_ov_fig_C_B_ij[::-1]
#    mean_cj_fig_G_A = mean_cj_fig_G_A_ji + mean_cj_fig_G_A_ij[::-1]
#    mean_cj_fig_G_B = mean_cj_fig_G_B_ji + mean_cj_fig_G_B_ij[::-1]
#    mean_cv_fig_K_A = mean_cv_fig_K_A_ji + mean_cv_fig_K_A_ij[::-1]
#    mean_cv_fig_K_B = mean_cv_fig_K_B_ji + mean_cv_fig_K_B_ij[::-1]

            #mean_ov_fig_C_A, mean_ov_fig_C_B, mean_cj_fig_G_A, mean_cj_fig_G_B, mean_cv_fig_K_A, mean_cv_fig_K_B)



def graph():
    (ovb_rate_low, ovb_rate_medium, ovb_rate_high, mean_A_chosen_cj, mean_B_chosen_cj, mean_low_cv, mean_medium_cv,mean_high_cv) = result_firing_rate()
     #mean_ov_fig_C_A, mean_ov_fig_C_B, mean_cj_fig_G_A, mean_cj_fig_G_B, mean_cv_fig_K_A, mean_cv_fig_K_B)
    X_axis = np.arange(0, 2.0, 0.0005)
    #X2_axis = ["1B: 20A", "1B: 16A", "1B: 12A", "1B: 8A", "1B: 4A", "4B: 1A", "8B: 1A", "12B: 1A", "16B: 1A", "20B: 1A"]
    bokeh.plotting.output_notebook()
    #figure_3 = bokeh.plotting.figure(title="Figure 3", plot_width=700, plot_height=700)
    figure_4_A = bokeh.plotting.figure(title="Figure 4 A", plot_width=700, plot_height=700)
    figure_4_E = bokeh.plotting.figure(title="Figure 4 E", plot_width=700, plot_height=700)
    figure_4_I = bokeh.plotting.figure(title="Figure 4 I", plot_width=700, plot_height=700)
    #figure_4_C = bokeh.plotting.figure(title="Figure 4 C", plot_width=700, plot_height=700)
    #figure_4_G = bokeh.plotting.figure(title="Figure 4 G", plot_width=700, plot_height=700)
    #figure_4_K = bokeh.plotting.figure(title="Figure 4 K", plot_width=700, plot_height=700)

    #figure_3.circle(x=np.arange(0,20), y=np.arange(0,20), color='blue', size = 10)
    figure_4_A.multi_line([X_axis, X_axis, X_axis], [ovb_rate_low, ovb_rate_medium, ovb_rate_high] , color =['red', "green", "blue"])
    figure_4_E.multi_line([X_axis, X_axis] , [mean_A_chosen_cj, mean_B_chosen_cj], color =['red', "blue"])
    figure_4_I.multi_line([X_axis, X_axis, X_axis] , [mean_low_cv, mean_medium_cv, mean_high_cv], color =['red', "green", "blue"])
    #figure_4_C.diamond([X2_axis], [mean_ov_fig_C_A], color ='red', size = 1)
    #figure_4_C.circle([X2_axis], [mean_ov_fig_C_B], color = "blue", size =1)
    #figure_4_G.diamond([X2_axis], [mean_cj_fig_G_A], color ='red', size = 1)
    #figure_4_G.circle([X2_axis], [mean_cj_fig_G_B], color="blue", size=1)
    #figure_4_K.diamond([X2_axis], [mean_cv_fig_K_A], color ='red', size = 1)
    #figure_4_K.circle([X2_axis], [mean_cv_fig_K_B], color="blue", size=1)


    #bokeh.plotting.show(figure_3)
    bokeh.plotting.show(figure_4_A)
    bokeh.plotting.show(figure_4_E)
    bokeh.plotting.show(figure_4_I)
    #bokeh.plotting.show(figure_4_C)
    #bokeh.plotting.show(figure_4_G)
    #bokeh.plotting.show(figure_4_K)

graph()






    ## Notes

"""
* In equations 11, 12, 13, 15, 16 and 17, _pyr and _in suffixes have been corrected to _rec_in and
_pyr_in respectively.
* In equation 17, I_gaba_rec_i has been corrected into I_gaba_rec_I.
* In the parameters, F has been corrected into f
"""
