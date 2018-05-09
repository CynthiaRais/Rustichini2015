import pathos

import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


ΔA, ΔB, n = (0, 10), (0, 20), 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

def compute_fig7_data(model_class=Model, w_p=1.82, hysteresis=True):
    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1, w_p=w_p,
                        hysteresis=hysteresis)

    filename='data/fig7_{}[{}]{}.pickle'.format(hysteresis, n, model.desc)
    return run_model(model, offers, history_keys=('r_2',), filename=filename)


if __name__ == '__main__':
    def aux(args):
        compute_fig7_data(hysteresis=args['hysteresis'], model_class=args['model_class'])

    runs=[{'model_class': model, 'hysteresis': hysteresis} for hysteresis in [True]
                                             for model in [Model, ReplicatedModel]]
    pool = pathos.multiprocessing.Pool()
    pool.map(aux, runs)
