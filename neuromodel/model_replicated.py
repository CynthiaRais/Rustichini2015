import numpy as np
from scipy.stats import norm

from . import model


class ReplicatedModel(model.Model):
    """This model reproduce the behavior of the Matlab code obtained from Aldo Rustichini
    and Camillo Padoa-Schioppa.
    """

    def firing_rate_I(self, phi_I):  # 2
        """Compute the update of  the firing rate of interneurons (eq. 2)

        In this equation, `self.τ_ampa` should be `self.τ_gaba`.
        """
        return ((-self.r['I'] + phi_I) / self.τ_ampa) * self.dt

    def range_adaptation(self, x, x_min, x_max):
        """Compute the range adaptation of a juice quantity (eq. 20)

        Here, `x / x_max` should be `(x - xmin) / (x_max - xmin)`. In all (#FIXME: verify!)
        published results in the article, x_min == 0, so this makes no difference.
        """
        return x / x_max

    def I_nmda_rec(self, i):  # 11
        """Compute the recurrent NMDA current for CJA and CJB cells (eq. 11)"""
        assert i in ('1', '2')
        j = '2' if i == '1' else '1' # j != i
        return (-self.N_E * self.f * self.J_nmda_rec_pyr * (self.δ_J_nmda[i] * self.w_p * self.S_nmda[i] + self.w_m * self.S_nmda[j])
                - self.N_E * (1 - 2 * self.f) * self.J_nmda_rec_pyr * self.w_m * self.S_nmda['3'])


class QuantitativelyReplicatedModel(ReplicatedModel):
    """This model quantitatively reproduces the behavior of the Matlab code obtained from
    Aldo Rustichini and Camillo Padoa-Schioppa. This allow to verify that the exact same sequence
    of values is generated by the two code.

    Only the way to produce Gaussian noise has been changed from `MatlabModel`, to generate the
    same random sequence as the Matlab code. As this is much less efficient, this model is much
    slower than ReplicatedModel, and therefore is only used in `test_replicated.py`.
    """

    def η(self):
        """Compute η, white noise with unit noise.

        Here we replace the call to `np.random.normal()` by the one below, as it generates
        the same random sequence as the Matlab code, if you also replace `randn` by the equivalent
        `norminv(rand,0,1)` in the Matlab code. In the Matlab code, the same random seed should be
        used as the one given to the `Model` class.

        The code below is significantly slower than `np.random.normal()`, taking as much as 80% of
        the model computing time.
        """
        return norm.ppf(self.random.rand()) # compatible with Matlab noise, much slower.
