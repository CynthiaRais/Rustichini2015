"""This save the behavior of the model on disk, to be commited to the git repository,
and used to check if the behavior of the model change from version to version of the code.

This script should be run each time the behavior of the model is *purposefully* (and precisely)
altered.
"""

import dotdot
from neuromodel import Model

def generate_testdata(seed=0):
    """Return the history of a run"""
    model = Model(ΔA=(0, 20), ΔB=(0, 20), random_seed=seed)

    results = []
    for x_a, x_b in [(1, 1), (4, 4), (3, 6), (1, 10)]:
        model.one_trial(x_a, x_b)
        data = model.trial_history.export(keys=['r_ovb', 'r_1', 'r_3', 'r_I'])
        sparse_data = {k: v[::100] for k, v in data.items()}
        results.append(sparse_data)

    return results


if __name__ == '__main__':
    import pickle

    testdata = generate_testdata(0)
    with open('data/testdata.pickle', 'wb') as f:
        pickle.dump(testdata, f)
