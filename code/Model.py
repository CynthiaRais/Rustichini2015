#-*- coding: utf-8 -*-

import numpy
import math
import itertools


#firing rate_1


def firing_pyr_cells(r_i, phy, τ_ampa, dt):
    r_i += ((- r_i + phy) / τ_ampa )* dt
    return r_i


#open channels AMPA_3


def channel_ampa(s_ampa, τ_ampa, r_i, dt):
    s_ampa += ((- s_ampa / τ_ampa) +r_i) * dt
    return s_ampa


#open channels NMDA_4


def channel_nmda(s_nmda, s_nmda_1, τ_nmda, γ, r_i, dt):
    s_nmda += ((-s_nmda / τ_nmda) + (1 - s_nmda_1) * γ * r_i)*dt
    return s_nmda


#formula Abbott and Chance_6


def input_output(i_syn, c, g, i):
    phy = (c * i_syn - i) /\
          (1 - numpy.exp(- g * (c * i_syn - i)))
    return phy


##Pyramidal cells
#Input current_7


def input_current(i_ampa_ext, i_ampa_rec, i_nmda_rec, i_gaba_rec, i_stim) :
    i_syn = i_ampa_ext + i_ampa_rec + i_nmda_rec + i_gaba_rec + i_stim
    return i_syn

#current for AMPA exterieur_8


def i_ampa_ext(j_ampa_ext_pyr, τ_ampa, connect_ext, r_ext, i_nu):
    i_ampa_ext_pyr = -j_ampa_ext_pyr * τ_ampa * connect_ext * r_ext + i_nu
    return i_ampa_ext_pyr


#current for AMPA recurrent_9


def i_ampa_rec(n_e, f, j_ampa_pyr, w_p, w_m, s_ampa_1, s_ampa_2, s_ampa_3):
    i_ampa_rec = - n_e * f * j_ampa_pyr * (w_p * s_ampa_1 + w_m * s_ampa_2) \
                - n_e * (1 - 2 * f) * j_ampa_pyr * w_m * s_ampa_3
    return i_ampa_rec

#S_AMPA : liste des S_AMPA?


#current for AMPA recurrent_10


def i_ampa_rec_3(n_e, f, j_ampa_pyr, s_ampa_1, s_ampa_2, s_ampa_3):
    i_ampa_rec_ns = - n_e * f * j_ampa_pyr * (s_ampa_1 + s_ampa_2) \
                   - n_e * (1 - 2 * f) * j_ampa_pyr * s_ampa_3
    return i_ampa_rec_ns


#current for NMDA recurrent_11


def i_nmda_rec(n_e, f, j_nmda_pyr, δ_j_ndma, w_p, s_nmda_1, w_m, s_nmda_2, s_nmda_3):
    i_nmda_rec_pyr = - n_e * f * j_nmda_pyr * δ_j_ndma * (w_p * s_nmda_1 + w_m * s_nmda_2) \
                   - n_e * (1 - 2 * f) * j_nmda_pyr * w_m * s_nmda_3
    return i_nmda_rec_pyr


# current for NMDA recurrent 3 _ 12


def i_nmda_rec_3(n_e, f, j_nmda_pyr, s_nmda_1, s_nmda_2, s_nmda_3):
    i_nmda_rec_ns = - n_e * f * j_nmda_pyr * (s_nmda_1 + s_nmda_2) \
                   - n_e *(1 - 2 * f) * j_nmda_pyr * s_nmda_3
    return i_nmda_rec_ns


#current GABA recurrent_13


def i_gaba_rec(n_i, j_gaba_pyr, δ_j_gaba, s_gaba):
    i_gaba_rec_ns = - n_i * j_gaba_pyr * δ_j_gaba * s_gaba
    return i_gaba_rec_ns



##For interneurons
#firing rate_2


def firing_rate_I(r_i,phy, τ_gaba, dt):
    r_i =((- r_i + phy) / τ_gaba) * dt
    return r_i


def channel_gaba(s_gaba, τ_gaba, r_i, dt):
    s_gaba_in = (- s_gaba / τ_gaba + r_i) * dt
    return s_gaba_in


#current AMPA exterieur_14


def i_ampa_ext_i(j_ampa_ext_i, τ_ampa, connect_ext, r_ext, i_nu):
    i_ampa_ext_in = - j_ampa_ext_i * τ_ampa * connect_ext * r_ext + i_nu
    return i_ampa_ext_in

#current AMPA recurrent_15



def i_ampa_rec_i(n_e, f, j_ampa_rec_i, s_ampa_1, s_ampa_2, s_ampa_3):
    i_ampa_rec_in = - n_e * f * j_ampa_rec_i * (s_ampa_1 + s_ampa_2) \
                   - n_e * (1 - 2 * f) * j_ampa_rec_i * s_ampa_3
    return i_ampa_rec_in


#current NMDA recurrent_16


def i_nmda_rec_i(n_e, f, j_nmda_rec_i, s_nmda_1, s_nmda_2, s_nmda_3) :
    i_nmda_rec_in = - n_e * f * j_nmda_rec_i * (s_nmda_1 + s_nmda_2) \
                   - n_e * (1 - 2 * f) * j_nmda_rec_i * s_nmda_3
    return i_nmda_rec_in


#current GABA recurrent_17


def i_gaba_rec_i(n_i, j_gaba_rec_i, s_gaba):
    i_gaba_rec_in = - n_i * j_gaba_rec_i * s_gaba
    return i_gaba_rec_in


##Noise_18
#Gaussian white noise


def nu(dt):
    normale = numpy.random.normal(0,1 / math.sqrt(dt))
    return normale

#White noise


def white_noise(i_nu, τ_ampa, σ, dt):
    i_nu = ((- i_nu + nu(dt) * math.sqrt(τ_ampa * (σ ** 2))) / τ_ampa) * dt
    return i_nu

##Offer cells
#equation 19


def i_stim(j_ampa_input, δ_j_hl, δ_j_stim, τ_ampa, r_ov):
    i_stim_ov = - j_ampa_input * δ_j_hl * δ_j_stim * τ_ampa * r_ov
    return i_stim_ov


#equation 20,21,22,23


def firing_ov_cells(x, xmin, x_max, t, a, b, c, d, r_o, Δ_r):
    x_i=(x - xmin)/(x_max - xmin)
    g = (1 / (1 + numpy.exp(- (t - a) / b))) * 1 / (1 + numpy.exp((t - c) / d))
    function = g / numpy.max(g)
    r_ov = r_o + Δ_r * function * x_i
    print('r_ov', r_ov)
    return r_ov



#parameters

n_e = 1.600
n_i = 400
connect_ext = 800
f = 0.15
r_ext = 3.0
τ_ampa = 2
τ_nmda = 100
τ_gaba = 5
j_ampa_ext_pyr = - 0.1123
j_ampa_rec_pyr = - 0.0027
j_nmda_rec_pyr = - 0.00091979
j_gaba_rec_pyr = 0.0215
j_ampa_ext_i = - 0.0842
j_ampa_rec_i = - 0.0022
j_nmda_rec_i = - 0.00083446
j_gaba_rec_i = 0.0180
γ = 0.641
σ = 0.020

i_e = 125
g_e = 0.16
c_e = 310
i_i = 177
g_i = 0.087
c_i = 615

t_offer = 1000
r_o = 0 #ou 6
Δ_r = 8
a = t_offer + 175
b = 30
c = t_offer + 400
d = 100
j_ampa_input = 30 * j_ampa_ext_pyr
w_p = 1.75
w_m = 1 - f * (w_p - 1) / (1 - f)

nb_stimulation = 4000
dt = 0.5

δ_j_hl_cj_a, δ_j_hl_cj_b = 1, 1
δ_j_stim_cj_a, δ_j_stim_cj_b = 2, 1
δ_j_gaba_cj_a, δ_j_gaba_cj_b, δ_j_gaba_ns = 1, 1, 1
δ_j_nmda_cj_a, δ_j_nmda_cj_b = 1, 1

i_stim_cv, i_stim_ns = 0, 0


def ov_a_cells(T_i, x_a, xmin, x_max):

    r_ov_a = firing_ov_cells(x_a, xmin, x_max, T_i, a, b, c, d, r_o, Δ_r)
    return r_ov_a


def ov_b_cells(T_i, x_b, xmin, x_max):

    r_ov_b = firing_ov_cells(x_b, xmin, x_max, T_i, a, b, c, d, r_o, Δ_r)
    return r_ov_b


def cj_a_cells(r_i_cj_a, s_ampa_cj_a, s_nmda_cj_a, s_gaba_cj_a, i_nu_cj_a,
               s_ampa_cj_b, s_ampa_ns, s_nmda_cj_b, s_nmda_ns, r_i_cv_cells, r_ov_a):

    s_ampa_cj_a = channel_ampa(s_ampa_cj_a, τ_ampa, r_i_cj_a, dt)
    s_nmda_cj_a = channel_nmda(s_nmda_cj_a, s_nmda_cj_a, τ_nmda, γ, r_i_cj_a, dt)
    s_gaba_cj_a = channel_gaba(s_gaba_cj_a, τ_gaba, r_i_cv_cells, dt)
    s_cj_a = [s_ampa_cj_a, s_nmda_cj_a, s_gaba_cj_a]

    i_nu_cj_a = white_noise(i_nu_cj_a, τ_ampa, σ, dt)
    i_ampa_ext_cj_a = i_ampa_ext(j_ampa_ext_pyr, τ_ampa, connect_ext, r_ext, i_nu_cj_a)
    i_ampa_rec_cj_a = i_ampa_rec(n_e, f, j_ampa_rec_pyr, w_p, w_m, s_ampa_cj_a, s_ampa_cj_b, s_ampa_ns)
    i_nmda_rec_cj_a = i_nmda_rec(n_e, f, j_nmda_rec_pyr, δ_j_nmda_cj_a, w_p, s_nmda_cj_a, w_m, s_nmda_cj_b, s_nmda_ns)
    i_gaba_rec_cj_a = i_gaba_rec(n_i, j_gaba_rec_pyr, δ_j_gaba_cj_a, s_gaba_cj_a)
    i_stim_cj_a = i_stim(j_ampa_input, δ_j_hl_cj_a, δ_j_stim_cj_a, τ_ampa, r_ov_a)
    i_syn_cj_a = input_current(i_ampa_ext_cj_a, i_ampa_rec_cj_a, i_nmda_rec_cj_a, i_gaba_rec_cj_a, i_stim_cj_a)

    phy_cj_a = input_output(i_syn_cj_a, c_e, g_e, i_e)
    r_i_cj_a = firing_pyr_cells(r_i_cj_a, phy_cj_a, τ_ampa, dt)
    return r_i_cj_a, s_cj_a, i_nu_cj_a


def cj_b_cells(r_i_cj_b, s_ampa_cj_b, s_nmda_cj_b, s_gaba_cj_b, i_nu_cj_b,
               s_ampa_cj_a, s_ampa_ns, s_nmda_cj_a, s_nmda_ns, r_i_cv_cells, r_ov_b):

    s_ampa_cj_b = channel_ampa(s_ampa_cj_b, τ_ampa, r_i_cj_b, dt) # equation 3
    s_nmda_cj_b = channel_nmda(s_nmda_cj_b, s_nmda_cj_a, τ_nmda, γ, r_i_cj_b, dt) # equation 4
    s_gaba_cj_b = channel_gaba(s_gaba_cj_b, τ_gaba, r_i_cv_cells, dt) # equation 5
    s_cj_b = [s_ampa_cj_b, s_nmda_cj_b, s_gaba_cj_b]

    i_nu_cj_b = white_noise(i_nu_cj_b, τ_ampa, σ, dt) # equation 18
    i_ampa_ext_cj_b = i_ampa_ext(j_ampa_ext_pyr, τ_ampa, connect_ext, r_ext, i_nu_cj_b) # equation 8
    i_ampa_rec_cj_b = i_ampa_rec(n_e, f, j_ampa_rec_pyr, w_p, w_m, s_ampa_cj_b, s_ampa_cj_a, s_ampa_ns) # equation 9
    i_nmda_rec_cj_b = i_nmda_rec(n_e, f, j_nmda_rec_pyr, δ_j_nmda_cj_b, w_p, s_nmda_cj_a, w_m, s_nmda_cj_b, s_nmda_ns) # equation 11
    i_gaba_rec_cj_b = i_gaba_rec(n_i, j_gaba_rec_pyr, δ_j_gaba_cj_b, s_gaba_cj_b) # equation 13
    i_stim_cj_b = i_stim(j_ampa_input, δ_j_hl_cj_b, δ_j_stim_cj_b, τ_ampa, r_ov_b) # equation 19
    i_syn_cj_b = input_current(i_ampa_ext_cj_b, i_ampa_rec_cj_b, i_nmda_rec_cj_b, i_gaba_rec_cj_b, i_stim_cj_b) #equation 7
    print('i_syn_cj_b', i_syn_cj_b)
    phy_cj_b = input_output(i_syn_cj_b, c_e, g_e, i_e) #equation 6
    r_i_cj_b = firing_pyr_cells(r_i_cj_b, phy_cj_b, τ_ampa, dt) #equation 1
    return r_i_cj_b, s_cj_b, i_nu_cj_b


def ns_cells(r_i_ns, s_ampa_ns, s_nmda_ns, s_gaba_ns, i_nu_ns,
             s_ampa_cj_a, s_ampa_cj_b, s_nmda_cj_a, s_nmda_cj_b, r_i_cv_cells):

    s_ampa_ns = channel_ampa(s_ampa_ns, τ_ampa, r_i_ns, dt) #equation 3
    s_nmda_ns = channel_nmda(s_nmda_ns, s_nmda_cj_a, τ_nmda, γ, r_i_ns, dt) #equation 4
    s_gaba_ns = channel_gaba(s_gaba_ns, τ_gaba, r_i_cv_cells, dt) #equation 5
    s_ns = [s_ampa_ns, s_nmda_ns, s_gaba_ns]

    i_nu_ns = white_noise(i_nu_ns, τ_ampa, σ, dt) #equation 18
    i_ampa_ext_ns = i_ampa_ext(j_ampa_ext_pyr, τ_ampa, connect_ext, r_ext, i_nu_ns) #equation 8
    i_ampa_rec_ns = i_ampa_rec_3(n_e, f, j_ampa_rec_pyr, s_ampa_cj_a, s_ampa_cj_b, s_ampa_ns) #equation 10
    i_nmda_rec_ns = i_nmda_rec_3(n_e, f, j_nmda_rec_pyr, s_nmda_cj_a, s_nmda_cj_b, s_nmda_ns) #equation 12
    i_gaba_rec_ns = i_gaba_rec(n_i, j_gaba_rec_pyr, δ_j_gaba_ns, s_gaba_ns) #equation 13

    i_syn_ns = input_current(i_ampa_ext_ns, i_ampa_rec_ns, i_nmda_rec_ns, i_gaba_rec_ns, i_stim_ns) #equation 7
    phy_ns = input_output(i_syn_ns, c_e, g_e, i_e) #equation 6
    r_i_ns = firing_pyr_cells(r_i_ns, phy_ns, τ_ampa, dt) #equation 1
    return r_i_ns, s_ns, i_nu_ns



def cv_cells(r_i_cv_cells, s_gaba_cv, i_nu_cv,
             s_ampa_cj_a, s_ampa_cj_b, s_ampa_ns,
             s_nmda_cj_a, s_nmda_cj_b, s_nmda_ns):

    s_gaba_cv = channel_gaba(s_gaba_cv, τ_gaba, r_i_cv_cells, dt) #equation 5

    i_nu_cv = white_noise(i_nu_cv, τ_ampa, σ, dt)  #equation 18
    i_ampa_ext_cv = i_ampa_ext_i(j_ampa_ext_i, τ_ampa, connect_ext, r_ext, i_nu_cv)  #equation 14
    i_ampa_rec_cv = i_ampa_rec_i(n_e, f, j_ampa_rec_i, s_ampa_cj_a, s_ampa_cj_b, s_ampa_ns) #equation 15
    i_nmda_rec_cv = i_nmda_rec_i(n_e, f, j_nmda_rec_i, s_nmda_cj_a, s_nmda_cj_b, s_nmda_ns) #equation 16
    i_gaba_rec_cv = i_gaba_rec_i(n_i, j_gaba_rec_i, s_gaba_cv) #equation 17
    i_syn_cv_cells = input_current(i_ampa_ext_cv, i_ampa_rec_cv, i_nmda_rec_cv, i_gaba_rec_cv, i_stim_cv) #equation 7

    phy_cv_cells = input_output(i_syn_cv_cells, c_i, g_i, i_i) #equation 6
    r_i_cv_cells += firing_rate_I(r_i_cv_cells, phy_cv_cells, τ_gaba, dt) #equation 2
    return r_i_cv_cells, s_gaba_cv, i_nu_cv


def quantity_juice():
    ''' random choice of juice quantity '''

    x_a = numpy.random.randint(0, 20)
    x_b = numpy.random.randint(0, 20)
    while x_a == 0 and x_b == 0:
        p = numpy.random.random()
        if p < 0.5:
            x_a = numpy.random.randint(0, 20)
        else:
            x_b = numpy.random.randint(0, 20)
    xmin = numpy.minimum(x_a, x_b)
    x_max = numpy.maximum(x_a, x_b)
    return x_a, x_b, xmin, x_max


def one_trial(T, r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
              i_nu_cj_a, i_nu_cj_b, i_nu_ns, i_nu_cv,
              s_cj_a, s_cj_b, s_ns, s_gaba_cv):

    x_a, x_b, xmin, x_max = quantity_juice()
    for T_i in T :
        r_ov_a = ov_a_cells(T_i, x_a, xmin, x_max)
        r_ov_b = ov_b_cells(T_i, x_b, xmin, x_max)

        r_i_cj_a, s_cj_a, i_nu_cj_a = cj_a_cells(r_i_cj_a, s_cj_a[0], s_cj_a[1], s_cj_a[2],
                                                 i_nu_cj_a, s_cj_b[0], s_ns[0], s_cj_b[1], s_ns[1], r_i_cv_cells, r_ov_a)
        r_i_cj_b, s_cj_b, i_nu_cj_b = cj_b_cells(r_i_cj_b, s_cj_b[0], s_cj_b[1], s_cj_b[2],
                                                 i_nu_cj_b, s_cj_a[0], s_ns[0], s_cj_a[1], s_ns[1], r_i_cv_cells, r_ov_b)
        r_i_ns, s_ns, i_nu_ns = ns_cells(r_i_ns, s_ns[0], s_ns[1], s_ns[2],
                                         i_nu_ns,s_cj_a[0], s_cj_b[0], s_cj_a[1], s_cj_b[1], r_i_cv_cells)
        r_i_cv_cells, s_gaba_cv, i_nu_cv = cv_cells(r_i_cv_cells, s_gaba_cv,
                                                    i_nu_cv, s_cj_a[0], s_cj_b[0], s_ns[0], s_cj_a[1], s_cj_b[1], s_ns[1])

    if r_i_cj_a > r_i_cj_b :
        choice = 'choice A'
    elif r_i_cj_a < r_i_cj_b:
        choice = 'choice B'
    else :
        return "error in choice"

    offer = '{0} A : {1} B'.format(x_a, x_b)
    r_ov_liste = [r_ov_a, r_ov_b]
    r_i_liste = r_i_cj_a, r_i_cj_b, r_i_cv_cells
    return  offer, choice, r_ov_liste, r_i_liste

def session():
    r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells = 0, 0, 0, 0
    i_nu_cj_a, i_nu_cj_b, i_nu_ns, i_nu_cv = 0, 0, 0, 0
    s_cj_a, s_cj_b, s_ns = [1, 1, 1], [1, 1, 1], [1, 1, 1]
    s_gaba_cv = 1

    result = []

    for i in range(4000):
        T = numpy.arange(-500, 1000, dt)
        result.append([one_trial(T, r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
                                i_nu_cj_a, i_nu_cj_b, i_nu_ns, i_nu_cv,
                                s_cj_a, s_cj_b, s_ns, s_gaba_cv)])

#        offer, choice, r_ov_liste, r_i_liste = one_trial(T, r_i_cj_a, r_i_cj_b, r_i_ns, r_i_cv_cells,
#                                                        i_nu_cj_a, i_nu_cj_b, i_nu_ns, i_nu_cv,
#                                                        s_cj_a, s_cj_b, s_ns, s_gaba_cv)

    return result

print(session())

#
#    for result_i in result :
#        ordered_list = itertools.groupby(result, )

#    low, medium, high = [], [], []

#        for i in range(20):
#            for j in range(20):
#                if offer == '{0} A : {1} B'.format(i, j):




#def analysis():
#    result = session()
#    if x_b <  20/3 :
#        low.append([r_ov_b, r_i_cj_b, r_i_cv_cells])
#    elif x_b > 20*2/3 :
#        high.append([r_ov_b, r_i_cj_b, r_i_cv_cells])
#    else :
#        medium.append([r_ov_b, r_i_cj_b, r_i_cv_cells])




#res = {}
#res[(1,5)] = []
#res[(1,5)].append()

#def step(dt)
#    t+=dt


