import unittest
import numpy as np

import dotdot
from neuromodel import Model, DataAnalysis



class AnalysisTest(unittest.TestCase):

    def test_means(self):
        """Test of the mean firing rate of ovb cells for Figure 4 graphs."""
        model = Model(ΔA=(0, 15), ΔB=(0, 15),
                      random_seed=0, t_exp=120*0.0005, σ_η=0)
        model.history.trials[(1, 1)].append({'r_ova': [0, 1, 2, 3],
                                                  'r_ovb': [5, 6, 7, 8]})
        model.history.trials[(1, 1)].append({'r_ova': [3, 2, 1, 0],
                                                  'r_ovb': [5, 6, 7, 8]})
        model.history.trials_choice[(1, 1, 'A')].append({'r_ova': [0, 1, 2, 3],
                                                         'r_ovb': [5, 6, 7, 8]})
        model.history.trials_choice[(1, 1, 'A')].append({'r_ova': [3, 2, 1, 0],
                                                         'r_ovb': [5, 6, 7, 8]})

        analysis = DataAnalysis(model)
        self.assertEqual(list(analysis.means[(1, 1)]['r_ova']), [1.5, 1.5, 1.5, 1.5])
        self.assertEqual(list(analysis.means[(1, 1)]['r_ovb']), [5, 6, 7, 8])

        self.assertTrue(np.all(analysis.means[(1, 1)]['r_ova'] == analysis.means_choice[(1, 1, 'A')]['r_ova']))
        self.assertTrue(np.all(analysis.means[(1, 1)]['r_ovb'] == analysis.means_choice[(1, 1, 'A')]['r_ovb']))



if __name__ == "__main__":
    unittest.main()
