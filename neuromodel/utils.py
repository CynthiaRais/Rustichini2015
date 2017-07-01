from functools import wraps
import inspect
import pickle
import time
import os

from tqdm import tqdm

from .data_analysis import DataAnalysis


def human_duration(seconds):
    """Return a human readable duration as minutes/seconds"""
    return "{}:{:} minutes and {:>04.1f} seconds".format(seconds // 60, seconds % 60)


def autoinit(init_fun):
    """Autoinitialize the instances members from the list of arguments given to __init__

    >>> class Model:
    ...     @autoinit
    ...     def __init__(self, α, γ=0.9):
    ...         pass
    >>> m = Model(0.5)
    >>> m.α, m.γ
    (0.5, 0.9)
    """
    argnames, _, _, defaults = inspect.getargspec(init_fun)

    @wraps(init_fun)
    def wrapper(self, *args, **kwargs):
        # args
        for name, arg in zip(argnames[1:], args):
            setattr(self, name, arg)
        # keywords args
        for name, arg in kwargs.items():
            setattr(self, name, arg)
        # defaults args
        for name, default in zip(reversed(argnames), reversed(defaults)):
            if not hasattr(self, name): # keyword value given?
                setattr(self, name, default)
        init_fun(self, *args, **kwargs)

    return wrapper


def run_model(model, offers, filename=None, opportunistic=True, verbose=True,
              history_keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
    """Run a model on a set of offers.

    :param opportunistic:  if it finds a file named `filename`, it will load it rather
                           than running the model.
    """
    if opportunistic and os.path.isfile(filename): # load from disk.
        if verbose:
            print('Loading results of {} from disk: {}.'.format(model.__class__.__name__,
                                                                    filename))
        with open(filename, 'rb') as f:
            analysis = pickle.load(f)

    else: # compute from scratch.
        start_time = time.time()
        if verbose:
            print('Computing results of {}.'.format(model.__class__.__name__))

        model.history.keys = history_keys # only save specific data

        for x_A, x_B in tqdm(offers.offers):
            model.one_trial(x_a=x_A, x_b=x_B)

        analysis = DataAnalysis(model)
        analysis.clear_history()

        if verbose:
            print('Done! (took {})'.format(human_duration(time.time() - start_time)))
            print('Saving results to {}.'.format(filename))
        with open(filename, 'wb') as f:
             pickle.dump(analysis, f)


    return analysis
