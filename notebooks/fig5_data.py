import pathos

import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


# configuring offers
ΔA, ΔB, n = 20, 20, 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)


def compute_fig5_data(model_class=Model, δ_J_stim=(1, 1),
                      δ_J_nmda=(1, 1), δ_J_gaba=(1, 1, 1), desc=''):
    """Compute the data for Figure 5."""
    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1,
                        range_A=offers.range_A, range_B=offers.range_B,
                        δ_J_stim=δ_J_stim, δ_J_gaba=δ_J_gaba, δ_J_nmda=δ_J_nmda)

    filename_suffix = '_replicate' if model_class is ReplicatedModel else ''
    filename='data/fig5_{}[{}]{}.pickle'.format(desc, n, filename_suffix)
    run_model(model, offers, history_keys=(), filename=filename)

if __name__ == '__main__':

    def aux(args):
        compute_fig5_data(model_class=args['model_class'],
                          δ_J_stim=args['δ_J_stim'], δ_J_nmda=args['δ_J_nmda'],
                          δ_J_gaba=args['δ_J_gaba'], desc=args['desc'])

    runs=[{'model_class': ReplicatedModel, 'desc': 'AMPA',
           'δ_J_stim': (2, 1), 'δ_J_nmda': (1, 1),    'δ_J_gaba': (1, 1, 1)},
          {'model_class': ReplicatedModel, 'desc': 'NMDA',
           'δ_J_stim': (1, 1), 'δ_J_nmda': (1.05, 1), 'δ_J_gaba': (1, 1, 1)},
          {'model_class': ReplicatedModel, 'desc': 'GABA',
           'δ_J_stim': (1, 1), 'δ_J_nmda': (1, 1),    'δ_J_gaba': (1, 1.02, 1)},
          {'model_class':           Model, 'desc': 'AMPA',
           'δ_J_stim': (2, 1), 'δ_J_nmda': (1, 1),    'δ_J_gaba': (1, 1, 1)},
          {'model_class':           Model, 'desc': 'NMDA',
           'δ_J_stim': (1, 1), 'δ_J_nmda': (1.05, 1), 'δ_J_gaba': (1, 1, 1)},
          {'model_class':           Model, 'desc': 'GABA',
           'δ_J_stim': (1, 1), 'δ_J_nmda': (1, 1),    'δ_J_gaba': (1, 1.02, 1)}]

    pool = pathos.multiprocessing.Pool()
    pool.map(aux, runs)
