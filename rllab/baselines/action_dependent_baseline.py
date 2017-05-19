from rllab.baselines.base import Baseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.product import Product
import numpy as np


class ActionDependentBaseline(Baseline):
    def __init__(self, env_spec):
        # FIXME(cathywu) hack to handle sudoku env and multiagent point envs
        if isinstance(env_spec.action_space, Product):
            self.nactions = len(env_spec.action_space.components)
            self.stride = env_spec.action_space.components[0].flat_dim
        elif isinstance(self.env.action_space, Box):
            self.nactions = env_spec.action_space.flat_dim
            self.stride = 1
        self.action_dependent = True

    @overrides
    def get_param_values(self, **tags):
        return [b.get_param_values() for b in self._sub_baselines]

    @overrides
    def set_param_values(self, val, **tags):
        for k in range(self.nactions):
            self._sub_baselines[k].set_param_values(val, **tags)

    def _features(self, path):
        return [b._features(path, idx=k) for k, b in
                enumerate(self._sub_baselines)]

    @overrides
    def fit(self, paths):
        for k in range(self.nactions):
            self._sub_baselines[k].fit(paths, idx=range(k * self.stride, k * self.stride + self.stride))

    @overrides
    def predict(self, path):
        return np.array(
            [b.predict(path, idx=range(k * self.stride, k * self.stride + self.stride))
             for k, b in enumerate(self._sub_baselines)])
