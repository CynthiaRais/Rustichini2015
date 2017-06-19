import numpy as np

class Offers:

    def __init__(self, ΔA=20, ΔB=20, n=4000, random_seed=1, range_A=None, range_B=None):
        self.random = np.random.RandomState(seed=random_seed)

        self.n = n
        self.ΔA, self.ΔB = ΔA, ΔB

        self.generate_offers()

        self.range_A = range_A
        if self.range_A is None:
            self.range_A = min(x_A for x_A, x_B in self.offers), max(x_A for x_A, x_B in self.offers)
        self.range_B = range_B
        if self.range_B is None:
            self.range_B = min(x_B for x_A, x_B in self.offers), max(x_B for x_A, x_B in self.offers)

    def generate_offers(self):
        """Generate the list of offers (x_A, x_B) of juice quantity"""
        self.offers = []
        while len(self.offers) < self.n:
            self.offers += self.generate_block()
        self.random.shuffle(self.offers)
        self.offers = self.offers[:self.n]


    def generate_block(self):
        """Generate a block of every of every possible offer"""
        block = [(x_A, x_B) for x_A in range(0, self.ΔA + 1)
                            for x_B in range(0, self.ΔB + 1)][1:] # remove (0, 0)
        return block


class SpecificOffers(Offers):

    def __init__(self, specific_offers=((0, 1), (1, 10)), **kwargs):
        self.specific_offers = specific_offers
        super().__init__(**kwargs)

    def generate_offers(self):
        offer_idx = range(len(self.specific_offers))
        chosen_idx = np.random.choice(offer_idx, size=self.n, replace=True)
        self.offers = [self.specific_offers[idx] for idx in chosen_idx]
