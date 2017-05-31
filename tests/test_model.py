import unittest
import numpy as np

import dotdot
from code import Model

class ModelTest(unittest.TestCase):

    def test_firing_pyr(self):
        '''Test of firing rate of pyramidal cells'''
        firing = 0
        for t in np.arange(0,1.5, 0.0005):
            firing = Model.firing_pyr_cells(firing, 31, 0.002, 0.0005)
            self.assertLess(firing, 30)
            self.assertGreaterEqual(firing, 5)

    def test_firing_interneurons(self):
        '''Test of firing rate of interneurons'''
        firing_interneurons = 0
        for t in np.arange(0, 1.5, 0.0005):
            firing_interneurons = Model.firing_rate_I(firing_interneurons, 31, 0.002, 0.0005)
            self.assertLess(firing_interneurons, 20)
            self.assertGreater(firing_interneurons, 10)


    def test_channels(self):
        '''Test of fraction of channels opened for pyramidal cells
        r_i est compris entre 5 et 30'''
        s_ampa = 0
        s_nmda = 0
        for t in np.arange(0, 1.5, 0.0005):
            for r_i in range(5, 30):
                s_ampa = Model.channel_ampa(s_ampa, 0.002, r_i, 0.0005)
                s_nmda = Model.channel_nmda(s_nmda, 0.01, 0.641, r_i, 0.0005)
                self.assertGreaterEqual(s_ampa,0)
                self.assertLessEqual(s_ampa, 1)
                self.assertGreaterEqual(s_nmda, 0)
                self.assertLessEqual(s_nmda, 1)

    def test_channel_gaba(self):
        '''Test of fraction of channel nmda opened for pyramidal cells
        r_i est compris entre 5 et 30'''
        s_gaba = 0
        for t in np.arange(0,1.5, 0.0005):
            for r_i in np.arange(0, 1.5, 0.0005):
                s_gaba = Model.channel_gaba(s_gaba, 0.005, 20, 0.0005)
                self.assertGreaterEqual(s_gaba, 0)
                self.assertLessEqual(s_gaba, 1)

    def test_Φ(self):
        '''Test of formula Abbott and Chance'''
        for i in np.arange(0, 0.6, 0.0005):
            test_phi_pyr = Model.Φ(i, 310, 0.16, 125)
            test_phi_interneurons = Model.Φ(i, 615, 0.087, 177)
            self.assertGreaterEqual(40, test_phi_pyr)
            self.assertLessEqual(-40, test_phi_pyr)
            self.assertGreaterEqual(21, test_phi_interneurons)
            self.assertLessEqual(-21, test_phi_interneurons)

    def test_intensity(self):
        '''test of current of pyramidal cells'''
        I_ampa_min = Model.I_ampa_rec(1600, 0.15, -0.0027, 1.75, (1- 0.15 *(1.75 - 1) /(1 - 0.15)), 0, 0, 0)
        I_ampa_max = Model.I_ampa_rec(1600, 0.15, -0.0027, 1.75, (1- 0.15 *(1.75 - 1) /(1 - 0.15)), 1, 1, 1)
        I_nmda_min = Model.I_nmda_rec(1600, 0.15, -0.00091979, 1, 1.75, 0, (1- 0.15 *(1.75 - 1) /(1 - 0.15)), 0, 0)
        I_nmda_max = Model.I_nmda_rec(1600, 0.15, -0.00091979, 1, 1.75, 1, (1- 0.15 *(1.75 - 1) /(1 - 0.15)), 1, 1)
        I_gaba_min = Model.I_gaba_rec(400, 0.0180, 0, 0)
        I_gaba_max = Model.I_gaba_rec(400, 0.0180, 1, 1)

        for i in np.arange(0,1):
            for j in np.arange(0,1):
                for k in np.arange(0,1):
                    I_ampa_rec_test = Model.I_ampa_rec(1600, 0.15, -0.0027, 1.75, (1- 0.15 *(1.75 - 1) /(1 - 0.15)), i, j, k)
                    I_nmda_rec_test = Model.I_nmda_rec(1600, 0.15, -0.00091979, 1, 1.75, i,(1- 0.15 *(1.75 - 1) /(1 - 0.15)), j, k)
                    I_gaba_rec_test = Model.I_gaba_rec(400, 0.0180, 1, i)
                    self.assertLessEqual(I_ampa_rec_test,I_ampa_max)
                    self.assertGreaterEqual(I_ampa_rec_test, I_ampa_min)
                    self.assertLessEqual(I_nmda_rec_test, I_nmda_max)
                    self.assertGreaterEqual(I_nmda_rec_test, I_nmda_min)
                    self.assertLessEqual(I_gaba_rec_test, I_gaba_max)
                    self.assertGreaterEqual(I_gaba_rec_test, I_gaba_min)


    def test_firing_ov(self):
        '''Test of firing rate of OV cells'''

        for t in np.arange(0, 1.5, 0.0005):
            firing_ov_min = Model.firing_ov_cells(0, 0, 20, t, 0.03, 0.1, 0, 8)
            self.assertEqual(firing_ov_min, 0)
            firing_ov_max = Model.firing_ov_cells(20, 0, 20, t, 0.03, 0.1, 0, 8)
            self.assertLessEqual(firing_ov_max, 8)
            self.assertGreaterEqual(firing_ov_max,0)

    def test_one_trial(self):
        '''Test one trial'''

        m = Model()
        for i in range(10):
            (choice, np.max(mean_ov_b), np.max(r_i_cj_a_tot), np.max(r_i_cj_b_tot), np.max(r_i_ns_tot),
             np.max(r_i_cv_cells_tot), np.max(s_ampa_cj_b_tot), np.max(s_nmda_b_tot), np.max(s_gaba_b_tot),
             np.max(s_gaba_cv_tot),
             np.min(mean_ov_b), np.min(r_i_cj_a_tot), np.min(r_i_cj_b_tot), np.min(r_i_ns_tot),
             np.min(r_i_cv_cells_tot),
             np.min(s_ampa_cj_b_tot), np.min(s_nmda_b_tot), np.min(s_gaba_b_tot), np.min(s_gaba_cv_tot)) = m.one_trial(0, 1, [0, 1], [20, 20], 0,
                                                                                                                0, 0, 0, 0,
                                                                                                                 0,0,0,0,
                                                                                                                    [0, 0, 0], [ 0, 0, 0], [0, 0, 0], 0)
            self.assertEqual(choice, 'no choice')
            self.assertEqual(np.max(mean_ov_b), 0.4000000000000003 )
            self.assertEqual(np.max(r_i_cj_a_tot), nan)
            self.assertEqual(np.max(r_i_cj_b_tot), nan)
            self.assertEqual(np.max(r_i_ns_tot), nan)
            self.assertEqual(np.max(r_i_cv_cells_tot), nan)
            self.assertEqual(np.max(s_ampa_cj_b_tot), nan)
            self.assertEqual(np.max(s_nmda_b_tot), nan)
            self.assertEqual(np.max(s_gaba_b_tot), nan)
            self.assertEqual(np.max(s_gaba_cv_tot), nan)
            self.assertEqual(np.min(mean_ov_b), 8.9306182607322952e-11)
            self.assertEqual(np.max(r_i_cv_cells_tot), nan)
            self.assertEqual(np.min(r_i_cj_a_tot), nan)
            self.assertEqual(np.min(r_i_cj_b_tot), nan)
            self.assertEqual(np.min(r_i_ns_tot), nan)
            self.assertEqual(np.min(s_ampa_cj_b_tot), nan)
            self.assertEqual(np.min(s_nmda_b_tot), nan)
            self.assertEqual(np.min(s_gaba_b_tot), nan)
            self.assertEqual(np.min(s_gaba_cv_tot), nan)

    if __name__ == '__main__':
        unittest.main()
