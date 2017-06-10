import numpy as np

class Offers:

    def __init__(self, ΔA=20, ΔB=20, n=4000, random_seed=1):
        self.random = np.random.RandomState(seed=random_seed)

        self.n = n
        self.ΔA, self.ΔB = ΔA, ΔB

        # generate offers, list offers (x_A, x_B) of juice quantity
        self.offers = []
        while len(self.offers) < self.n:
            self.offers += self.generate_block()
        self.random.shuffle(self.offers)
        self.offers = self.offers[:self.n]

        self.range_A = min(x_A for x_A, x_B in self.offers), max(x_A for x_A, x_B in self.offers)
        self.range_B = min(x_B for x_A, x_B in self.offers), max(x_B for x_A, x_B in self.offers)

    def generate_block(self):
        """Generate a block of every of every possible offer"""
        block = [(x_A, x_B) for x_A in range(0, self.ΔA + 1)
                            for x_B in range(0, self.ΔB + 1)][1:] # remove (0, 0)
        return block
