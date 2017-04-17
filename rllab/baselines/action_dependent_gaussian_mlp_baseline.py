import numpy as np

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.baselines.action_dependent_baseline import ActionDependentBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.overrides import overrides


class ActionDependentGaussianMLPBaseline(ActionDependentBaseline,
                                         Baseline, Parameterized,
                                         Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(ActionDependentGaussianMLPBaseline, self).__init__(env_spec)

        self._sub_baselines = [
            GaussianMLPBaseline(env_spec, subsample_factor=subsample_factor,
                                num_seq_inputs=num_seq_inputs,
                                action_dependent=True,
                                regressor_args=regressor_args) for _ in
            range(self.nactions)]
