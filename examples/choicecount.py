import dotdot
from neuromodel import Model


choices = {'A': 0, 'B': 0}
model = Model(random_seed=0)

for _ in range(1000):
    model.one_trial(1, 4)
    choices[model.choice] += 1

print(choices)
