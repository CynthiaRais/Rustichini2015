from functools import wraps
import inspect


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
