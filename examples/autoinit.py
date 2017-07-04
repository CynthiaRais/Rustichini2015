import dotdot
from neuromodel.utils import autoinit


if __name__ == '__main__':

    class Model:
        @autoinit
        def __init__(self, α, γ=0.9):
            pass

    m = Model(0.5)
    print(m.α, m.γ)
