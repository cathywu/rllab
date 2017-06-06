import numpy as np

from rllab.core.serializable import Serializable
from rllab.baselines.action_dependent_baseline import ActionDependentBaseline
# from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.overrides import overrides


class ActionDependentGaussianMLPBaseline(ActionDependentBaseline,
                                         Serializable):
    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            mix_fraction=1.0,
            include_time=True,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(ActionDependentGaussianMLPBaseline, self).__init__(env_spec)

        self._sub_baselines = [
            GaussianMLPBaseline(env_spec, subsample_factor=subsample_factor,
                                action_dependent=True, scope="vf%s" % idx,
                                include_time=include_time,
                                mix_fraction=mix_fraction,
                                regressor_args=regressor_args) for idx in
            range(self.nactions)]
