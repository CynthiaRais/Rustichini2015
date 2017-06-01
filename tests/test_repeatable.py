"""
Test if the run are repeatable, i.e., successive runs yield the same results, given a fixed
random seed.
"""

import unittest
import numpy as np
import array

import dotdot
from code import Model, settings


def array2list(data):
    """Replace np.array and array to lists in a complex data structure.

    This function was written to easily check the equality of histories.
    """
    if isinstance(data, (list, tuple, array.array)):
        return [array2list(e) for e in data]
    elif isinstance(data, dict):
        return {k: array2list(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


class RepeatableTests(unittest.TestCase):
    """Test repeatability"""

    def test_repeatable(self):
        """Test that two 50-steps runs yield the same history"""

        setting = settings.Presetting(ΔA=20, ΔB=20)
        quantity_a, quantity_b, x_min_list, x_max_list = setting.quantity_juice()

        def run(seed):
            """Return the history of a run"""
            model = Model(x_min_list=[0, 0], x_max_list=[20, 20], random_seed=seed)
            return model.one_trial(1, 10)

        self.assertEqual(run(0), run(0))


if __name__ == '__main__':
    unittest.main()
