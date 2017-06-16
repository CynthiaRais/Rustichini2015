"""
Test if the run are repeatable, i.e., successive runs yield the same results, given a fixed
random seed.
"""

import unittest
import array

import numpy as np
import scipy.io

import dotdot
from neuromodel import Model


VERBOSE=False

def access_data(datamat, key):
    return list(datamat['history'][key][0][0][0])

class ReplicableTests(unittest.TestCase):
    """Test replicability"""

    def test_replicable(self, verbose=VERBOSE, x_a=1,x_b=0):
        """Test that the matlab code remains """
        K = 6101

        datamat_ref = scipy.io.loadmat('testdata_ref.mat')
        datamat     = scipy.io.loadmat('testdata{}A{}B.mat'.format(x_a, x_b))
        model = Model(range_A=[0, 15], range_B=[0, 15], random_seed=0, t_exp=(100+K)*0.0005, σ_eta=0,
                      full_log=True)
        model.one_trial(x_a, x_b)

        if verbose:
            for key_mat, key_py in [('sampa1', 'S_ampa_1'), ('nu1', 'r_1'), ('Iamparec1', 'I_ampa_rec_1'),
                                    ('phi1', 'phi_1'), ('Isyn1' , 'Isyn_1'), ('Istim1', 'I_stim_1')]:
                print('{} [{}]'.format(key_py, key_mat))
                print('matlab: {}'.format(', '.join('{: 8.4f}'.format(e) for e in access_data(datamat, key_mat)[:K])))
                print('python: {}'.format(', '.join('{: 8.4f}'.format(e) for e in getattr(model.trial_history, key_py)[:K])))


        np.testing.assert_almost_equal(access_data(datamat, 'sampa1')[:K], model.trial_history.S_ampa_1[:K])
        np.testing.assert_almost_equal(access_data(datamat, 'nu1')[:K], model.trial_history.r_1[:K])
        np.testing.assert_almost_equal(access_data(datamat, 'nuI')[:K], model.trial_history.r_I[:K])

    def test_replication(self):
        for (x_a,x_b) in [(1,0), (1,10), (4,1), (5,10)]:
            self.test_replicable(x_a = x_a, x_b = x_b)


if __name__ == '__main__':
    unittest.main()
