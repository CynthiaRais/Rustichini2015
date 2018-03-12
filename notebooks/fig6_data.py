import pathos

import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


ΔA, ΔB, n = (0, 20), (0, 20), 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

def compute_fig6_data(w_p, model_class=Model):
    """

    If the result filename already exists, the computation will be skipped.
    :param model_class:  set to ReplicatedModel if you want to replicate the published figures.
                         set to Model to use the 'corrected' model, as described in the article.
    """
    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1, w_p=w_p)

    filename_suffix = '_replicate' if model_class is ReplicatedModel else ''
    filename='data/fig6_{}[{}]{}.pickle'.format(w_p, n, filename_suffix)
    return run_model(model, offers, history_keys=('r_2', 'r_I'), filename=filename)


if __name__ == '__main__':
        def aux(args):
            compute_fig6_data(args['w_p'], model_class=args['model_class'])

        runs=[{'model_class': model, 'w_p': w_p} for w_p in [1.55, 1.70, 1.85]
              for model in [Model, ReplicatedModel]]
        pool = pathos.multiprocessing.Pool()
        pool.map(aux, runs)
