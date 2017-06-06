from rllab.baselines.base import Baseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.action_dependent_baseline import ActionDependentBaseline
from rllab.misc.overrides import overrides
import numpy as np


class ActionDependentLinearFeatureBaseline(ActionDependentBaseline):
    def __init__(self, env_spec, reg_coeff=1e-5,
                 mix_fraction=1.0,  include_time=True, **kwargs):
        super(ActionDependentLinearFeatureBaseline, self).__init__(env_spec)
        self._sub_baselines = [LinearFeatureBaseline(env_spec,
                                                     reg_coeff=reg_coeff,
                                                     action_dependent=True,
                                                     stride=self.stride,
                                                     include_time=include_time,
                                                     mix_fraction=mix_fraction,
                                                     ) for _ in range(
            self.nactions)]
