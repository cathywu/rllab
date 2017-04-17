from rllab.baselines.base import Baseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.overrides import overrides
import numpy as np


class ActionDependentLinearFeatureBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        self.nactions = env_spec.action_space.shape[0]
        self._sub_baselines = [LinearFeatureBaseline(env_spec,
                     reg_coeff=reg_coeff) for _ in range(self.nactions)]
        self._reg_coeff = reg_coeff

    @overrides
    def get_param_values(self, **tags):
        return [b.get_param_values() for b in self._sub_baselines]

    @overrides
    def set_param_values(self, val, **tags):
        for idx, v in enumerate(val):
            self._sub_baselines[idx]._coeffs = v

    def _features(self, path):
        return [b._features(path, idx=k) for k, b in
                enumerate(self._sub_baselines)]

    @overrides
    def fit(self, paths):
        for k in range(self.nactions):
            self._sub_baselines[k].fit(paths, idx=k)

    @overrides
    def predict(self, path):
        return np.array([b.predict(path, idx=k) for k,b in enumerate(
                self._sub_baselines)])
