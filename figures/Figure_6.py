
# coding: utf-8

# # Figure 6

# In[ ]:


import dotdot
from neuromodel import Offers, Model, ReplicatedModel, run_model
import neuromodel.graphs

#%matplotlib notebook


# Like all figures in this replication effort, you can either employ the model that replicate the behavior of the Matlab code used to produce the figures in the original article, or the model that contains fixes to make it as close a the description in the original article.

# In[ ]:

smooth = 'mean'

for model_class in [Model, ReplicatedModel]:
# model_class=Model              # use the corrected model that matches the article's description
# model_class=ReplicatedModel  # use a model that can replicate published figures


# In[ ]:


    ΔA, ΔB, n = (0, 20), (0, 20), 4000
    offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)


    # In[ ]:


    def compute_fig6_data(model_class, w_p):
        model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1, w_p=w_p)

        filename = 'data/fig6_{}[{}]{}.pickle'.format(w_p, n, model.desc)
        return run_model(model, offers, history_keys=('r_2', 'r_I'),
                         smooth=smooth, filename=filename)


    # ## w+ = 1.55

    # In[ ]:


    analysis_1_55 = compute_fig6_data(model_class, 1.55)
    graph_1_55 = neuromodel.graphs.Graph(analysis_1_55)


    # In[ ]:


    data_6A = analysis_1_55.means_chosen_choice(key='r_2')
    figure_6A = graph_1_55.means_chosen_choice(data_6A, title='Figure 6A',
                                               y_range=(2, 16), y_ticks=(3, 6, 9, 12, 15))


    # In[ ]:


    data_6B = analysis_1_55.tuning_curve('r_2', (0.5, 1.0))
    figure_6B = graph_1_55.tuning_curve(data_6B, title='Figure 6B')


    # In[ ]:


    data_6C = analysis_1_55.means_lowmedhigh_AB('r_I')
    figure_6C = graph_1_55.means_lowmedhigh(data_6C, title='Figure 6C',
                                            y_range=(10, 17), y_ticks=(10, 12, 14, 16))


    # In[ ]:


    data_6D = analysis_1_55.means_chosen_value('r_I', time_window=(0, 0.5))
    figure_6D = graph_1_55.means_chosen_value(data_6D, title='Figure 6D',
                                              y_range=(10, 16), y_ticks=(10, 12, 14, 16))


    # ## w+ = 1.70

    # In[ ]:


    analysis_1_70 = compute_fig6_data(model_class, 1.70)
    graph_1_70 = neuromodel.graphs.Graph(analysis_1_70)


    # In[ ]:


    data_6E = analysis_1_70.means_chosen_choice(key='r_2')
    figure_6E = graph_1_70.means_chosen_choice(data_6E, title='Figure 6E',
                                               y_range=(0, 23), y_ticks=(0, 5, 10, 15, 20))


    # In[ ]:


    data_6F = analysis_1_70.tuning_curve('r_2', (0.5, 1.0))
    figure_6F = graph_1_70.tuning_curve(data_6F, title='Figure 6F')


    # In[ ]:


    data_6G = analysis_1_55.means_lowmedhigh_AB('r_I')
    figure_6G = graph_1_55.means_lowmedhigh(data_6G, title='Figure 6G',
                                            y_range=(10, 18), y_ticks=(10, 12, 14, 16, 18))


    # In[ ]:


    data_6H = analysis_1_70.means_chosen_value('r_I', time_window=(0, 0.5))
    figure_6H = graph_1_70.means_chosen_value(data_6H, title='Figure 6H',
                                              y_range=(10, 16.5), y_ticks=(10, 12, 14, 16))


    # ## w+ = 1.85

    # In[ ]:


    analysis_1_85 = compute_fig6_data(model_class, 1.85)
    graph_1_85 = neuromodel.graphs.Graph(analysis_1_85)


    # In[ ]:


    data_6I = analysis_1_85.means_chosen_choice(key='r_2')
    figure_6I = graph_1_85.means_chosen_choice(data_6I, title='Figure 6I',
                                               y_range=(0, 35), y_ticks=(0, 10, 20, 30))


    # In[ ]:


    data_6J = analysis_1_85.tuning_curve('r_2', (0.5, 1.0))
    figure_6J = graph_1_85.tuning_curve(data_6J, title='Figure 6J')


    # In[ ]:


    data_6K = analysis_1_85.means_lowmedhigh_AB('r_I')
    figure_6K = graph_1_85.means_lowmedhigh(data_6K, title='Figure 6K',
                                            y_range=(10, 19), y_ticks=(10, 12, 14, 16, 18))


    # In[ ]:


    data_6L = analysis_1_85.means_chosen_value('r_I', time_window=(0, 0.5))
    figure_6L = graph_1_85.means_chosen_value(data_6L, title='Figure 6L',
                                              y_range=(10, 17), y_ticks=(10, 12, 14, 16))


    #
