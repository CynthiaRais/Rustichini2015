"""This save the behavior of the model on disk, to be commited to the git repository,
and used to check if the behavior of the model change from version to version of the code.

This script should be run each time the behavior of the model is *purposefully* (and precisely)
altered.
"""

import dotdot
from code import Model, settings

def generate_testdata(seed=0):
    """Return the history of a run"""
    model = Model(x_min_list=[0, 0], x_max_list=[20, 20], random_seed=seed)
    model.one_trial(1, 10)

    results = []
    for x_a, x_b in [(1, 1), (4, 4), (3, 6), (1, 10)]:
        data = model.one_trial(x_a, x_b)
        sparse_data = [e[::100] for e in data]
        results.append(sparse_data)

    return results


if __name__ == '__main__':
    import pickle

    testdata = generate_testdata(0)
    with open('testdata.pickle', 'wb') as f:
        pickle.dump(testdata, f)
