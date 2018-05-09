import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


# configuring offers
ΔA, ΔB, n = (0, 20), (0, 20), 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

def compute_fig4_data(model_class=Model):
    """Compute the Figure 4 data used in the Figure_4 notebook.

    If the result filename already exists, the computation will be skipped.
    :param model_class:  set to ReplicatedModel if you want to replicate the published figures.
                         set to Model to use the 'corrected' model, as described in the article.
    """
    model = model_class(n=n, random_seed=1, ΔA=ΔA, ΔB=ΔB)

    filename='data/fig4[{}]{}.pickle'.format(n, model.desc)
    return run_model(model, offers, history_keys=('r_ovb', 'r_2', 'r_I'), filename=filename)


if __name__ == '__main__':
    compute_fig4_data(model_class=Model)
    compute_fig4_data(model_class=ReplicatedModel)
