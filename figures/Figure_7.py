
# coding: utf-8

# # Figure 7

# In[ ]:


import dotdot
from neuromodel import Model, ReplicatedModel, Offers, run_model, load_analysis
import neuromodel.graphs

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


A_COLOR = (229/255, 146/255, 149/255)
B_COLOR = (139/255, 145/255, 193/255)


# Like all figures in this replication effort, you can either employ the model that replicate the behavior of the Matlab code used to produce the figures in the original article, or the model that contains fixes to make it as close a the description in the original article.

# In[ ]:


smooth = 'mean'

for model_class in [Model, ReplicatedModel]:
# model_class=Model              # use the corrected model that matches the article's description
# model_class=ReplicatedModel  # use a model that can replicate published figures


    # In[ ]:


    ΔA, ΔB, n = (0, 10), (0, 20), 4000
    offers = Offers(ΔA=ΔA, ΔB=ΔB, n=n, random_seed=0)


    # In[ ]:


    def compute_fig7_data(model_class=Model, w_p=1.82, hysteresis=True, smooth='mean'):
        model = model_class(n=n, ΔA=ΔA, ΔB=ΔB, random_seed=1, w_p=w_p,
                            hysteresis=hysteresis)
        filename = 'data/fig7_{}[{}]{}.pickle'.format(hysteresis, n, model.desc)
        return run_model(model, offers, history_keys=('r_2',), filename=filename, smooth=smooth)


    # In[ ]:


    analysis_7CD = load_analysis('data/fig4[{}]{}.pickle'.format(n, model_class.desc))
    analysis_7CD.smooth = smooth
    graph_7CD = neuromodel.graphs.Graph(analysis_7CD)
    analysis_7EF = compute_fig7_data(model_class)
    graph_7EF = neuromodel.graphs.Graph(analysis_7EF)


    # In[ ]:


    # CJB firing across time
    data_7CD = analysis_7CD.choice_hysteresis(key='r_2')
    figure_7C = graph_7CD.means_previous_choice(data_7CD, title='Figure 7C', y_ticks =(0, 10, 20, 30, 40))


    # In[ ]:


    figure_7C_inset = graph_7CD.means_previous_choice(data_7CD, title='Figure 7C inset', size=200,
                          x_range=(-300, 100), x_ticks=(-300, -200, -100, 0, 100),
                          y_range=(3, 4.5), y_ticks =(3, 4,))


    # In[ ]:


    # % choice B
    fig = graph_7CD.regression_3D(analysis_7CD.regression_hysteresis(type='A.'), title='Figure 7D',
                                  show=False, azim=-61, elev=20, marker='s', point_color=A_COLOR)
    fig = graph_7CD.regression_3D(analysis_7CD.regression_hysteresis(type='B.'), title='Figure 7D',
                                  marker='o', point_color=B_COLOR, fig=fig)


    # In[ ]:


    # CJB firing across time
    data_7EF = analysis_7EF.choice_hysteresis(key='r_2')
    figure_7E = graph_7EF.means_previous_choice(data_7EF, title='Figure 7E', y_ticks =(0, 10, 20, 30, 40))


    # In[ ]:


    figure_7E_inset = graph_7EF.means_previous_choice(data_7EF, title='Figure 7E inset', size=200,
                          x_range=(-300, 100), x_ticks=(-300, -200, -100, 0, 100),
                          y_range=(2, 10), y_ticks =(2, 4, 6, 8, 10))


    # In[ ]:


    # % choice B
    fig = graph_7EF.regression_3D(analysis_7EF.regression_hysteresis(type='A.'), title='Figure 7F',
                                  show=False, azim=-33, elev=20, marker='s', point_color=A_COLOR)
    fig = graph_7EF.regression_3D(analysis_7EF.regression_hysteresis(type='B.'), title='Figure 7F',
                                  marker='o', point_color=B_COLOR, fig=fig)


    #
