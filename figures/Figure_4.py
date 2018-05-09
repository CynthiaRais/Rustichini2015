
# coding: utf-8

# # Figure 4

# In[ ]:


import dotdot
from neuromodel import Model, ReplicatedModel, Offers, run_model
import neuromodel.graphs

#%matplotlib notebook


# Like all figures in this replication effort, you can either employ the model that replicate the behavior of the Matlab code used to produce the figures in the original article, or the model that contains fixes to make it as close a the description in the original article.

# In[ ]:

smooth = 'mean'

# # model_class=Model              # use the corrected model that matches the article's description
# model_class=ReplicatedModel  # use a model that can replicate published figures

for model_class in [Model, ReplicatedModel]:


    # In[ ]:


    ΔA, ΔB, n = (0, 20), (0, 20), 4000
    offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)

    x_offers = ((1, 0), (20, 1), (16, 1), (12, 1), (8, 1), (4, 1), # specific offers for Figures 4C, 4G, 4K
                (1, 4), (1, 8), (1, 12), (1, 16), (1, 20), (0, 1))


    # In[ ]:


    def compute_fig4_data(model_class=Model):
        model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1)

        filename = 'data/fig4[{}]{}.pickle'.format(n, model.desc)
        return run_model(model, offers, history_keys=('r_ovb', 'r_2', 'r_I'),
                         smooth=smooth, filename=filename)


    # In[ ]:


    analysis = compute_fig4_data(model_class)
    graph = neuromodel.graphs.Graph(analysis)


    # In[ ]:


    data_4A = analysis.means_lowmedhigh_B('r_ovb')
    figure_4A = graph.means_lowmedhigh(data_4A, title='Figure 4A',
                                       y_range=(0, 7), y_ticks=(0, 2, 4, 6))


    # In[ ]:


    data_4B = analysis.tuning_curve('r_ovb', time_window=(0, 0.5))
    figure_4B = graph.tuning_curve(data_4B, title='Figure 4B')


    # In[ ]:


    figure_4C = graph.specific_set(x_offers, analysis.means_offers('r_ovb', x_offers, time_window=(0.0, 0.5)),
                                   analysis.percents('B', x_offers), y_range=(0, 5), title='Figure 4C')


    # In[ ]:


    figure_4D = graph.firing_offer_B(analysis.tuning_curve('r_ovb', time_window=(0.0, 0.5)), title='Figure 4D')


    # In[ ]:


    data_4E = analysis.means_chosen_choice(key='r_2')
    figure_4E = graph.means_chosen_choice(data_4E, title='Figure 4E',
                                          y_range=(0, 25), y_ticks=(0, 5, 10, 15, 20, 25))


    # In[ ]:


    data_4F = analysis.tuning_curve('r_2', time_window=(0.5, 1.0))
    figure_4F = graph.tuning_curve(data_4F, title='Figure 4F')


    # In[ ]:


    figure_4G = graph.specific_set(x_offers, analysis.means_offers('r_2', x_offers, time_window=(0.5, 1.0)),
                                   analysis.percents('B', x_offers), y_range=(0, 16), title='Figure 4G')


    # In[ ]:


    figure_4H = graph.firing_choice(analysis.tuning_curve('r_2', time_window=(0.5, 1.0)))


    # In[ ]:


    data_4I = analysis.means_lowmedhigh_AB('r_I')
    figure_4I = graph.means_lowmedhigh(data_4I, title='Figure 4I',
                                       y_range=(10, 18), y_ticks=(10, 12, 14, 16, 18))


    # In[ ]:


    figure_4J = graph.tuning_curve(analysis.tuning_curve('r_I', time_window=(0.0, 0.5)), title='Figure 4J')


    # In[ ]:


    figure_4K = graph.specific_set(x_offers, analysis.means_offers('r_I', x_offers, time_window=(0.0, 0.5)),
                                   analysis.percents('B', x_offers), y_range=(10, 17), title='Figure 4K')


    # In[ ]:


    data_4L = analysis.means_chosen_value('r_I', time_window=(0, 0.5))
    figure_4L = graph.means_chosen_value(data_4L, title='Figure 4L',
                                         y_range=(10, 17), y_ticks=(10, 12, 14, 16))


    #
