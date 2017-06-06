# 1:CJA, 2:CJB, 3:NS


class TrialHistory:

    def __init__(self):

        self.data = {name: [] for name in ['r_cja', 'r_cjb', 'r_ns', 'r_cv',
                                           'S_ampa_cja', 'S_ampa_cjb', 'S_ampa_ns',
                                           'S_ndma_cja', 'S_ndma_cjb', 'S_ndma_ns',
                                           'S_gaba']}


    def __getattr__(self, name):
        return self.data[name]

        # self.r_ova, self.r_ovb = [], []
        # self.r_cja, self.r_cjb = [], []
        # self.r_ns, self.r_cv   = [], []
