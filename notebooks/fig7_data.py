import pathos

import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


ΔA, ΔB, n = (0, 10), (0, 20), 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

def compute_fig7_data(model_class=Model, w_p=1.82):
    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1, w_p=1.82,
                        hysteresis=True)

    filename='data/fig7_{}[{}]{}.pickle'.format(w_p, n, model.desc)
    return run_model(model, offers, history_keys=('r_2'), filename=filename)


if __name__ == '__main__':
    pool = pathos.multiprocessing.Pool()
    pool.map(compute_fig7_data, [Model, ReplicatedModel])
