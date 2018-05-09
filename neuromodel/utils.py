import functools
#from functools import wraps
import inspect
import pickle
import time
import os
import copy
import multiprocessing

from tqdm import tqdm

from .history import History
from .data_analysis import DataAnalysis


def human_duration(seconds):
    """Return a human readable duration as minutes/seconds"""
    seconds = int(seconds)
    return "{:d} minutes and {:d} seconds".format(seconds // 60, seconds % 60)


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

    @functools.wraps(init_fun)
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


def run_trial(offer, model):
    #print('run_trial {}'.format(offer))
    x_A, x_B = offer
    trial_history = model.one_trial(x_a=x_A, x_b=x_B)
    model.history.reset()
    return trial_history


def load_analysis(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run_model(model, offers, filename=None, opportunistic=True, verbose=True,
              parallel=False, preprocess=True, smooth='savgol',
              history_keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
    """Run a model on a set of offers.

    :param opportunistic:  if it finds a file named `filename`, it will load it rather
                           than running the model.
    """
    if opportunistic and os.path.isfile(filename): # load from disk.
        if verbose:
            print('Loading results of {} from disk: {}.'.format(
                  model.__class__.__name__, filename))
        analysis = load_analysis(filename)
        analysis.smooth = smooth

    else: # compute from scratch.
        start_time = time.time()
        if verbose:
            print('Computing results of {}.'.format(model.__class__.__name__))

        model.history.keys = history_keys # only save specific data

        if parallel: # use multiple processes to compute faster. Some random sequences are shared.
            raise ValueError('Does not work yet.')

            history = History(model)
            history.keys = model.history.keys # only save specific data

            pool = multiprocessing.Pool()
            func = functools.partial(run_trial, model=model)
            for trial_history in tqdm(pool.imap_unordered(func, offers.offers), total=len(offers.offers)):
                history.add_trial(trial_history)

            model.history = history

        else: # sequential, slower. Required for proper hysteresis.
            for x_A, x_B in tqdm(offers.offers):
                model.one_trial(x_a=x_A, x_b=x_B)

        analysis = DataAnalysis(model, preprocess=preprocess, smooth=smooth)
        if preprocess:
            analysis.clear_history()

        if verbose:
            print('Done! (took {})'.format(human_duration(time.time() - start_time)))
            print('Saving results to {}.'.format(filename))
        with open(filename, 'wb') as f:
             pickle.dump(analysis, f)


    return analysis
