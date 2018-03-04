import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model


# configuring offers
ΔA, ΔB, n = 20, 20, 4000
offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)


def compute_fig5_data(model_class=Model, δ_J_stim=(1, 1),
                      δ_J_nmda=(1, 1), δ_J_gaba=(1, 1, 1), suffix=''):
    """Compute the data for Figure 5."""
    model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1,
                        range_A=offers.range_A, range_B=offers.range_B,
                        δ_J_stim=δ_J_stim, δ_J_gaba=δ_J_gaba, δ_J_nmda=δ_J_nmda)

    filename_suffix = '_replicate' if model_class is ReplicatedModel else ''
    filename='data/fig5_{}[{}]{}.pickle'.format(suffix, n, filename_suffix)
    return run_model(model, offers, history_keys=(), filename=filename)


if __name__ == '__main__':
    compute_fig5_data(model_class=ReplicatedModel, δ_J_stim=(2, 1), suffix='AMPA')
    compute_fig5_data(model_class=ReplicatedModel, δ_J_nmda=(1.05, 1), suffix='NMDA')
    compute_fig5_data(model_class=ReplicatedModel, δ_J_gaba=(1, 1.02, 1), suffix='GABA')

    compute_fig5_data(model_class=Model, δ_J_stim=(2, 1), suffix='AMPA')
    compute_fig5_data(model_class=Model, δ_J_nmda=(1.05, 1), suffix='NMDA')
    compute_fig5_data(model_class=Model, δ_J_gaba=(1, 1.02, 1), suffix='GABA')
