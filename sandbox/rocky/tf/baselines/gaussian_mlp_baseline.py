import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.baselines.base import Baseline
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.regressors.moving_target_regressor import MovingTargetRegressor
from sandbox.rocky.tf.misc import space_utils


class GaussianMLPBaseline(Baseline, Parameterized, Serializable):
    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            scope="vf",
            mix_fraction=1.0,
            include_time=True,
            action_dependent=False,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self.observation_space = env_spec.observation_space
        self.horizon = env_spec.horizon
        self.include_time = include_time
        self.nactions = env_spec.action_space.flat_dim
        self.action_dependent = action_dependent

        # Notice the self.nactions-1 implicitly assumes conditional independence
        # TODO(cathywu) implement a more general version for general
        # factorized policies
        self._regressor = MovingTargetRegressor(
            GaussianMLPRegressor(
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                input_shape=(self.feature_size,),
                output_dim=1,
                name=scope,
                **regressor_args
            ),
            mix_fraction=mix_fraction
        )

    def get_features(self, path, idx=None):
        obs = path["observations"]
        feats = [obs]
        if self.include_time:
            T = len(obs)
            feats.append(np.arange(T).reshape((-1, 1)) / float(self.horizon))
        if idx is not None:
            minus_idx = [x for x in range(self.nactions) if x != idx]
            a = path["actions"][:, minus_idx]
            feats.append(a)
        return np.concatenate(feats, axis=-1)

    @property
    def feature_size(self):
        obs_dim = space_utils.space_to_flat_dim(self.observation_space)
        fsize = obs_dim  # Is this 2 from obs + last_obs or o, o ** 2?
        if self.include_time:
            fsize += 1
        if self.action_dependent:
            fsize += self.nactions - 1
        return fsize

    @overrides
    def fit(self, paths, idx=None):
        all_feats = np.concatenate([self.get_features(p, idx=idx) for p in
                                    paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(all_feats, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path, idx=None):
        feats = self.get_features(path, idx=idx)
        return self._regressor.predict(feats).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def get_params_internal(self, **tags):
        return self._regressor.get_params_internal(**tags)
