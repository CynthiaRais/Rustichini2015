import numpy as np

class Presetting:
    def __init__(self,  n = 4000,
                        ΔA = 20,
                        ΔB = 20):
        self.n = n
        self.ΔA, self.ΔB = ΔA, ΔB
        self.quantity_a, self.quantity_b = [], []  # list of juice quantity A and B
        self.x_min_list, self.x_max_list = [], []  # list of minimum and maximum of juice A and B in a session
        self.result_one_trial = {}

    def quantity_juice(self):
        # random choice of juice quantity, ΔA = ΔB = [0, 20]
        for i in range(self.n):
            self.x_a = np.random.randint(0, self.ΔA + 1)
            self.x_b = np.random.randint(0, self.ΔB + 1)
            while self.x_a == 0 and self.x_b == 0:
                p = np.random.random()
                if p < 0.5:
                    self.x_a = np.random.randint(0, self.ΔA + 1)
                else:
                    self.x_b = np.random.randint(0, self.ΔB + 1)
            self.quantity_a.append(self.x_a)
            self.quantity_b.append(self.x_b)
        self.x_min_list = [np.min(self.quantity_a)] + [np.min(self.quantity_b)]
        self.x_max_list = [np.max(self.quantity_a)] + [np.max(self.quantity_b)]
        return self.quantity_a, self.quantity_b, self.x_min_list, self.x_max_list

    def result_trials(self):
        for i in range(self.ΔA + 1):
            for j in range(self.ΔB + 1):
                    self.result_one_trial[(i, j)] = []
        return self.result_one_trial
