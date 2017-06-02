# 1:CJA, 2:CJB, 3:NS


class TrialHistory:

    def __init__(self):

        self.r_ova, self.r_ovb = [], []
        self.r_cja, self.r_cjb = [], []
        self.r_ns, self.r_cv   = [], []

        self.S_ampa_cja, self.S_ampa_cjb, self.S_ampa_ns = [], [], []
        self.S_ndma_cja, self.S_ndma_cjb, self.S_ndma_ns = [], [], []
        self.S_gaba = []
