"""Test that the behavior of the model is the same than previous code versions

This makes explicit changes that alter the model behavior and ones that don't.
If you modify the behavior of the model, you should update this test.
"""

import unittest
import pickle

import dotdot
from code import Model, settings
from update_testdata import generate_testdata

class ReproducibleTests(unittest.TestCase):
    """Test reproducibility"""

    def test_repeatable(self):
        """Test that data generated with previous version of the code match the current one"""

        testdata = generate_testdata()
        with open('testdata.pickle', 'rb') as f:
            refdata = pickle.load(f)

        self.assertEqual(testdata, refdata)

if __name__ == '__main__':
    unittest.main()
