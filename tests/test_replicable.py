"""
Test if the run are repeatable, i.e., successive runs yield the same results, given a fixed
random seed.
"""

import unittest
import array

import numpy as np
import scipy.io

import dotdot
from neuromodel import MatlabModel


VERBOSE  = False  # useful output when mismatch are detected.
FULL_LOG = False  # checks every variable, not only r_ovb, r_2, r_3 and r_I
                  # (requires full_log matlab files)

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

    def aux_test_replicable(self, x_a=5, x_b=10, rtol=0.0, atol=1e-8, # rtol=1e-9, atol=0,
                                  verbose=VERBOSE, full_log=FULL_LOG):
        """Test that the matlab code remains """
        K = 6000

        datamat     = scipy.io.loadmat('data/testdata{}A{}B.mat'.format(x_a, x_b))
        seed_mat = access_data(datamat, 'seed')[0]
        assert seed_mat != 0, "Seed 0 is not supported, due to implementation differences between Numpy as Matlab."
        slicesize   = access_data(datamat, 'slice')[0]
        Kslice = K // slicesize


        model = MatlabModel(range_A=[0, 20], range_B=[0, 20], random_seed=seed_mat,
                            t_exp=(100+K)*0.0005, full_log=True)
        model.one_trial(x_a, x_b)

        key_compare =  ['r_ovb', 'r_2', 'r_3', 'r_I']

        if full_log:
            key_compare += ['r_ova', 'r_1',
                            'S_ampa_1', 'S_ampa_2', 'S_ampa_3',
                            'S_nmda_1', 'S_nmda_2', 'S_nmda_3',
                            'S_gaba',
                            'I_eta_1', 'I_eta_2', 'I_eta_3', 'I_eta_I',
                            'I_ampa_rec_3', 'I_ampa_rec_I',
                            'I_syn_1', 'I_syn_2', 'I_syn3', 'I_syn_I',
                            'I_stim_1', 'I_stim_2']

        if verbose:
            no_mismatch = []
            for key in key_compare:
                data_py  = getattr(model.trial_history, key)[:K:slicesize]
                data_mat = access_data(datamat, key)[:Kslice]
                k = first_mismatch(data_py, data_mat, rtol=rtol, atol=atol)
                if k is None:
                    no_mismatch.append(key)
                else:
                    k_prev = max(0, k - 3)
                    print('{}: t={}-{} (first mismatch t={})'.format(key, k_prev, k_prev+10, k))
                    print('matlab: {}'.format(', '.join('{: 14.10f}'.format(e) for e in data_mat[k_prev:k_prev+10])))
                    print('python: {}'.format(', '.join('{: 14.10f}'.format(e) for e in  data_py[k_prev:k_prev+10])))
            print('No mismatch for: {}'.format(', '.join(no_mismatch)))

        for key in key_compare:
            data_py  = getattr(model.trial_history, key)[:K:slicesize]
            data_mat = access_data(datamat, key)[:Kslice]
            if not np.allclose(data_mat, data_py, rtol=rtol, atol=atol):
                print(key)
                np.testing.assert_allclose(data_mat, data_py, rtol=rtol, atol=atol)


    def test_replication(self):
        for x_a in range(21):
            for x_b in range(21):
                if x_a != 0 or x_b != 0:
                    print('checking {}A{}B ...'.format(x_a, x_b))
                    self.aux_test_replicable(x_a=x_a, x_b=x_b)
        print('done!')


    def test_specific_offers(self, offers=[(1, 10)]):
        """Test specific offers"""
        for x_a, x_b in offers:
                if x_a != 0 or x_b != 0:
                    print('checking {}A{}B ...'.format(x_a, x_b))
                    self.aux_test_replicable(x_a=x_a, x_b=x_b)
        print('done!')


if __name__ == '__main__':
    unittest.main()
