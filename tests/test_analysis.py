import unittest
import numpy as np

import dotdot
from code import Model
from code import DataAnalysis


class ModelTest(unittest.TestCase):

    def test_mean_firing_rate_ovb(self):
        """Test of the mean firing rate of ovb cells for Figure 4 graphs."""
        model = Model(ΔA=15, ΔB=15, x_min_list=[0, 0], x_max_list=[15, 15],
                      random_seed=0, t_exp=120*0.0005, σ_eta=0)
        model.history.trials[(1, 1, 'A')].append({'r_ova': [0, 1, 2, 3],
                                                  'r_ovb': [5, 6, 7, 8]})
        model.history.trials[(1, 1, 'A')].append({'r_ova': [3, 2, 1, 0],
                                                  'r_ovb': [5, 6, 7, 8]})
        analysis = DataAnalysis(model.history)
        self.assertEqual(list(analysis.means[(1, 1, 'A')]['r_ova']), [1.5, 1.5, 1.5, 1.5])
        self.assertEqual(list(analysis.means[(1, 1, 'A')]['r_ovb']), [5, 6, 7, 8])


#        analysis.mean_firing_rate_ovb()

if __name__ == '__main__':
    unittest.main()
