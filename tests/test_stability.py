"""Test that the behavior of the model is the same than previous code versions

This makes explicit changes that alter the model behavior and ones that don't.
If you modify the behavior of the model, you should update this test.
"""

import unittest
import pickle

import dotdot
from code import Model, settings
from update_testdata import generate_testdata

class RepeatableTests(unittest.TestCase):
    """Test repeatability"""

    def test_repeatable(self):
        """Test that two 50-steps runs yield the same history"""

        testdata = generate_testdata()
        with open('testdata.pickle', 'rb') as f:
            refdata = pickle.load(f)

        self.assertEqual(testdata, refdata)

if __name__ == '__main__':
    unittest.main()
