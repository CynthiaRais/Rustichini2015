from functools import wraps
import inspect
import pickle

from .data_analysis import DataAnalysis


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


def run_model(model, offers, filename=None, opportunistic=True,
              history_keys=('r_1', 'r_2', 'r_3', 'r_I', 'r_ova', 'r_ovb')):
    """Run a model against a set of offers

    :param opportunistic:  if it finds a file named `filename`, it will load it rather
                           than running the model.
    """
    try:
        if not opportunistic:
            raise FileNotFoundError
        with open(filename, 'rb') as f:
            analysis = pickle.load(f)

    except FileNotFoundError:

        model.history.keys = history_keys # only save specific data

        for i, (x_A, x_B) in enumerate(offers.offers):
            if (i + 1) % 100 == 0:
                print('step {}'.format(i + 1))
            model.one_trial(x_a=x_A, x_b=x_B)

        analysis = DataAnalysis(model)
        analysis.clear_history()
        with open(filename, 'wb') as f:
             pickle.dump(analysis, f)

    return analysis
