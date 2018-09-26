
# coding: utf-8

# # Figure 5

# In[ ]:


import dotdot
from neuromodel import Model, ReplicatedModel, Offers, run_model
import neuromodel.graphs


# Like all figures in this replication effort, you can either employ the model that replicate the behavior of the Matlab code used to produce the figures in the original article, or the model that contains fixes to make it as close a the description in the original article.

# In[ ]:


for model_class in [Model, ReplicatedModel]:

    # Create offers.

    # In[ ]:


    ΔA, ΔB, n = (0, 20), (0, 20), 4000
    offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

    x_offers = ((1, 0), (20, 1), (16, 1), (12, 1), (8, 1), (4, 1), # specific offers for Figures 4C, 4G, 4K
                (1, 4), (1, 8), (1, 12), (1, 16), (1, 20), (0, 1))


    # In[ ]:


    def compute_fig5_data(model_class, δ_J_stim=(1, 1),
                          δ_J_nmda=(1, 1), δ_J_gaba=(1, 1, 1), desc=''):
        """Compute tshe data for Figure 5."""
        model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1,
                            δ_J_stim=δ_J_stim, δ_J_gaba=δ_J_gaba, δ_J_nmda=δ_J_nmda)

        filename = 'data/fig5_{}[{}]{}.pickle'.format(desc, n, model.desc)
        return run_model(model, offers, history_keys=(), filename=filename)


    # In[ ]:


    analysis_ampa = compute_fig5_data(model_class, δ_J_stim=(2, 1), desc='AMPA')
    graph_ampa = neuromodel.graphs.Graph(analysis_ampa)


    # In[ ]:


    graph_ampa.regression_3D(analysis_ampa.data_regression(dim='3D'), title='Figure 5A')


    # In[ ]:


    graph_ampa.regression_2D(analysis_ampa.data_regression(dim='2D'), title='Figure_5B')


    # In[ ]:


    analysis_nmda = compute_fig5_data(model_class, δ_J_nmda=(1.05, 1), desc='NMDA')
    graph_nmda = neuromodel.graphs.Graph(analysis_nmda)


    # In[ ]:


    graph_nmda.regression_2D(analysis_nmda.data_regression(dim='2D'), title='Figure 5C')


    # In[ ]:


    analysis_gaba = compute_fig5_data(model_class, δ_J_gaba=(1, 1.02, 1), desc='GABA')
    graph_gaba = neuromodel.graphs.Graph(analysis_gaba)


    # In[ ]:


    graph_gaba.regression_2D(analysis_gaba.data_regression(dim='2D'), title='Figure 5D')


    #
