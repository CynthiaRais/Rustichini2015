import dotdot
from neuromodel import SpecificOffers

offers = SpecificOffers(specific_offers=((1, 4), (1, 10)), range_A=(0, 20), range_B=(0, 20), n=10)
print(offers.offers)
