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
    return list(datamat['data'][key][0][0][0])

class ReplicableTests(unittest.TestCase):
    """Test replicability"""

    def test_replicable(self):
        """Test that the matlab code remains """

        K = 10

        datamat_ref = scipy.io.loadmat('testdata_ref.mat')
        datamat     = scipy.io.loadmat('testdata.mat')
        model = Model(x_min_list=[0, 0], x_max_list=[15, 15], random_seed=0, t_exp=0.01, σ_eta=0)
        model.one_trial(1, K//2)
        print(len(model.trial_history.r_cja))


        for key_mat, key_py in [('sampa1', 'S_ampa_cja'), ('nu1', 'r_cja')]:
            print('{} [{}]'.format(key_py, key_mat))
            print('matlab: {}'.format(', '.join('{: 8.4f}'.format(e) for e in access_data(datamat, key_mat)[:K])))
            print('python: {}'.format(', '.join('{: 8.4f}'.format(e) for e in getattr(model.trial_history, key_py)[:K])))


        for k in range(K):
            self.assertEqual(access_data(datamat, 'sampa1')[k], model.trial_history.S_ampa_cja[k])
            self.assertEqual(access_data(datamat, 'nu1')[k], model.trial_history.r_cja[k])

#        self.assertEqual(access_data(datamat, 'nu1'), access_data(datamat_ref, 'nu1'))


if __name__ == '__main__':
    # import scipy.io
    # print(scipy.io.loadmat('testdata.mat'))
    unittest.main()
