"""
Test if the run are repeatable, i.e., successive runs yield the same results, given a fixed
random seed.
"""

import unittest
import array

import numpy as np
import scipy.io

import dotdot
from code import Model


def access_data(datamat, key):
    return list(datamat['data']['s1'][0][0][0])

class ReplicableTests(unittest.TestCase):
    """Test repeatability"""

    def test_replicable(self):
        """Test that two 50-steps runs yield the same history"""

        datamat = scipy.io.loadmat('testdata.mat')
        model = Model(x_min_list=[0, 0], x_max_list=[15, 15], random_seed=0, t_exp=0.01, σ_eta=0)
        model.one_trial(1, 10)

#        self.assertEqual(access_data(datamat, 'sampa1'), model.trial_history.S_ampa_cja)
        self.assertEqual(access_data(datamat, 'nu1'), model.trial_history.r_cja)


if __name__ == '__main__':
    # import scipy.io
    # print(scipy.io.loadmat('testdata.mat'))
    unittest.main()
