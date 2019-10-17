# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pymc import MCMC

from .models import AbstractModel
from .models import BernoulliModel  # noqa pylint: disable=unused-import
from .models import ExponentialModel  # noqa pylint: disable=unused-import
from .models import GaussianModel  # noqa pylint: disable=unused-import
from .models import LognormalModel  # noqa pylint: disable=unused-import
from .models import PoissonModel  # noqa pylint: disable=unused-import
from .models import StudentModel  # noqa pylint: disable=unused-import

models = {cls.key(): cls for cls in AbstractModel.__subclasses__()}


class BABTest:

    def __init__(self, control, variant, model='student', verbose=True):
        """Init.

        :param np.array control: 1 dimensional array of observations for control group
        :param np.array variant: 1 dimensional array of observations for variant group
        :param string model: desired distribution to describe both groups, defaults to Student
        """
        assert control.ndim == 1
        assert variant.ndim == 1
        self.control = control
        self.variant = variant
        self.sampler = None
        if not isinstance(model, AbstractModel):
            try:
                model = models[model]
            except KeyError:
                raise KeyError('Unknown model - please select a model from {}'.format(models.keys()))

        self.model = model(self.control, self.variant)
        self.verbose = verbose

    def run(self, n_iter=110000, n_burn=10000, thin=1):
        """Run the Bayesian test.

        :param int n_iter: total number of MCMC iterations
        :param int n_burn: no tallying done during the first n_burn iterations - these samples will be forgotten
        :param int thin: variables will be tallied at intervals of this many iterations

        :return: None
        """
        self.model.setup((n_iter - n_burn) / thin)
        self.sampler = MCMC(self.model.stochastics)
        self.sampler.sample(iter=n_iter, burn=n_burn, thin=thin, progress_bar=self.verbose)

    def plot(self, n_bins=30):
        """Display the results of the test.

        :param int n_bins: number of bins in the histograms

        :return: None
        """
        self.model.plot(n_bins=n_bins)
