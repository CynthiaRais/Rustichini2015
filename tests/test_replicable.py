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


VERBOSE = True
FULL_LOG = False

def access_data(datamat, key):
    return list(datamat['history'][key][0][0][0])

def first_mismatch(a, b, rtol=1e-07, atol=1e-07):
    """Return the index of the first mismatch between a and b, None if none exists"""
    for k, (a_i, b_i) in enumerate(zip(a, b)):
        if not np.allclose(a_i, b_i, rtol=rtol, atol=atol):
            return k
    return None


class ReplicableTests(unittest.TestCase):
    """Test replicability"""

    def aux_test_replicable(self, x_a=5,x_b=10, rtol=1e-07, atol=1e-07,
                                  verbose=VERBOSE, full_log=FULL_LOG):
        """Test that the matlab code remains """
        K = 6000

        datamat     = scipy.io.loadmat('data/testdata{}A{}B.mat'.format(x_a, x_b))
        model = Model(range_A=[0, 20], range_B=[0, 20], random_seed=0, t_exp=(100+K)*0.0005, σ_eta=0,
                      full_log=True)
        model.one_trial(x_a, x_b)

        for k in range(K):
            for i in ['1', '2', '3', 'I']:
                assert getattr(model.trial_history, 'I_eta_{}'.format(i))[k] == 0.0

        key_compare = [ ('nuOV2', 'r_ovb'),
                       ('nu2', 'r_2'), ('nu3', 'r_3'), ('nuI', 'r_I')]

        if full_log:
            key_compare = [('nuOV1', 'r_ova'), ('nu1', 'r_1'),
                           ('sampa1', 'S_ampa_1'), ('sampa2', 'S_ampa_2'), ('sampa3', 'S_ampa_3'),
                           ('snmda1', 'S_nmda_1'), ('snmda2', 'S_nmda_2'), ('snmda3', 'S_nmda_3'),
                           ('sgaba', 'S_gaba'),
                           ('I_eta1', 'I_eta_1'), ('I_eta2', 'I_eta_2'), ('I_eta3', 'I_eta_3'), ('I_etaI', 'I_eta_I'),
                           ('Iamparec3', 'I_ampa_rec_3'), ('IamparecI', 'I_ampa_rec_I'),
                           ('Isyn1', 'Isyn_1'), ('Isyn2', 'Isyn_2'), ('Isyn3', 'Isyn_3'),
                           ('Istim1', 'I_stim_1'), ('Istim2', 'I_stim_2')]

        if verbose:
            no_mismatch = []
            for key_mat, key_py in key_compare:
                k = first_mismatch(access_data(datamat, key_mat)[:K], getattr(model.trial_history, key_py)[:K], rtol=rtol, atol=atol)
                if k is None:
                    no_mismatch.append(key_py)
                else:
                    k_prev = max(0, k - 3)
                    print('{} [{}]: t={}-{} (first mismatch t={})'.format(key_py, key_mat, k_prev, k_prev+10, k))
                    print('matlab: {}'.format(', '.join('{: 14.10f}'.format(e) for e in access_data(datamat, key_mat)[k_prev:k_prev+10])))
                    print('python: {}'.format(', '.join('{: 14.10f}'.format(e) for e in getattr(model.trial_history, key_py)[k_prev:k_prev+10])))
            print('No mismatch for: {}'.format(', '.join(no_mismatch)))

        for key_mat, key_py in key_compare:
            if not np.allclose(access_data(datamat, key_mat)[:K],
                               getattr(model.trial_history, key_py)[:K],
                               rtol=rtol, atol=atol):
                print(key_mat, key_py)
            np.testing.assert_allclose(access_data(datamat, key_mat)[:K],
                                       getattr(model.trial_history, key_py)[:K],
                                       rtol=rtol, atol=atol)

    def test_replication(self):
       for (x_a,x_b) in [(1, 0), (5, 10), (4, 1), (1, 10)]:
           self.aux_test_replicable(x_a = x_a, x_b = x_b)


if __name__ == '__main__':
    unittest.main()
