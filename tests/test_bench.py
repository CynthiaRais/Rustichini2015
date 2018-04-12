import dotdot
from neuromodel import Model, History, DataAnalysis, Offers

ΔA, ΔB, n = (0, 20), (0, 20), 500
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n)

w_p = 1.55
model = Model(n=n, ΔA=ΔA, ΔB=ΔB, w_p=w_p, random_seed=1)


def run_model(model, offers, data_keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
    """Run a model against a set of offers

    :param opportunistic:  if it finds a file named `filename`, it will load it rather
                           than running the model.
    """
    model.history = History(model, keys=data_keys)

    for i, (x_A, x_B) in enumerate(offers.offers[:25]):
        model.one_trial(x_a=x_A, x_b=x_B)

    analysis = DataAnalysis(model)

    return analysis

if __name__ == '__main__':
    run_model(model, offers, data_keys=('r_2', 'r_I'))
