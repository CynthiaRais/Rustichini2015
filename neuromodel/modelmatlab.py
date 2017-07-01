import numpy as np
from scipy.stats import norm

from . import model


class MatlabModel(model.Model):
    """This model reproduce the behavior of the Matlab code obtained from Aldo Rustichini
    and Camillo Padoa-Schioppa.
    """

    def η(self):
        """Compute η, white noise with unit noise."""
        return self.random.normal(0, 1)

    def I_η_update(self, j):  # 18
        """Compute the update to I_η, modelized by a Ornstein-Uhlenbeck process (eq. 18).

        Here, `np.sqrt(self.dt / self.τ_ampa)` should be `self.dt / np.sqrt(self.τ_ampa)`
        """
        return (-self.I_η[j] * (self.dt / self.τ_ampa) +
                self.η() * np.sqrt(self.dt / self.τ_ampa) * self.σ_η)

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


class MatlabModelReplicable(MatlabModel):
    """This model quantitatively reproduces the behavior of the Matlab code obtained from
    Aldo Rustichini and Camillo Padoa-Schioppa. Only the way to produce Gaussian noise has been
    changed from `MatlabModel`, to generate the same random sequence as the Matlab code.
    """

    def η(self):
        """Compute η, white noise with unit noise.

        Here we replace the call to `np.random.normal()` by the one below, as it generates
        the same random sequence as the Matlab code, if you also replace `randn` by the equivalent
        `norminv(rand,0,1)` in the Matlab code.

        The code below is significantly slower than `np.random.normal()`, taking as much as 80% of
        the model computing time.
        """
        return norm.ppf(self.random.rand()) # compatible with Matlab noise, much slower.
