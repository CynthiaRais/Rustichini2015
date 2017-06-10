import dotdot
from neuromodel import Offers, Model, run_model


ΔA, ΔB, n = 20, 20, 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=1)

def compute_fig4_data():
    model = Model(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=0,
                  range_A=offers.range_A, range_B=offers.range_B)

    filename='data/fig4[{}].pickle'.format(n)
    return run_model(model, offers, history_keys=('r_ovb', 'r_2', 'r_I'), filename=filename)


if __name__ == '__main__':
    compute_fig4_data()
