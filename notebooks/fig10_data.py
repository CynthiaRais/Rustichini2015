import pathos

import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


ΔA, ΔB, n = (0, 20), (0, 20), 1500
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

J_ampa_rec_in = -0.00198 # 0.9 * J_ampa_rec_in
J_nmda_rec_in = -0.000751014 # 0.9 * J_nmda_rec_in
J_gaba_rec_in = 0.0144 # 0.8 * J_gaba_rec_in

x_offers = ((1, 0), (20, 1), (16, 1), (12, 1), (8, 1), (4, 1), # specific offers for Figures 4C, 4G, 4K
            (1, 4), (1, 8), (1, 12), (1, 16), (1, 20), (0, 1))

def compute_fig10_data(model_class=Model, network='symmetric'):
    δ_J_stim = {'symmetric':  (1  , 1),
                'asymmetric': (1.2, 1)}[network]

    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1,
                        r_o=6, w_p=1.65, J_ampa_rec_in=J_ampa_rec_in,
                        J_nmda_rec_in=J_nmda_rec_in, J_gaba_rec_in=J_gaba_rec_in,
                        δ_J_stim=δ_J_stim)

    filename = 'data/fig10_{}[{}]{}.pickle'.format(network, n, model.desc)
    return run_model(model, offers, history_keys=('r_ovb', 'r_2', 'r_I'), filename=filename)


if __name__ == '__main__':
    def aux(args):
        compute_fig10_data(model_class=args['model_class'], network=args['network'])

    runs=[{'model_class': ReplicatedModel, 'network':  'symmetric'},
          {'model_class': ReplicatedModel, 'network': 'asymmetric'},
          {'model_class':           Model, 'network':  'symmetric'},
          {'model_class':           Model, 'network': 'asymmetric'}]

    pool = pathos.multiprocessing.Pool()
    pool.map(aux, runs)
