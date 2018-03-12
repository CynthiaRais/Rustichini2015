import pathos

import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


ΔA, ΔB, n = (0, 20), (0, 20), 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

def compute_fig9_data(model_class, w_p=1.75, ΔJ=30):
    """
    If the result filename already exists, the computation will be skipped.
    :param model_class:  set to ReplicatedModel if you want to replicate the published figures.
                         set to Model to use the 'corrected' model, as described in the article.
    """
    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1, w_p=w_p, ΔJ=ΔJ)
    filename = 'data/fig10_{}_{}[{}]{}.pickle'.format(w_p, ΔJ, n, model.desc)
    return run_model(model, offers, history_keys=('r_ovb', 'r_2', 'r_I'), filename=filename)

if __name__ == '__main__':
        def aux(args):
            compute_fig9_data(**args)

        runs=[{'model_class': model, 'w_p': w_p, 'ΔJ': ΔJ}
              for w_p, ΔJ in [(1.75, 30), (1.85, 30), (1.75, 15)]
              for model in [Model, ReplicatedModel]]
        pool = pathos.multiprocessing.Pool()
        pool.map(aux, runs)
