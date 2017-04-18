import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor


class GaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            action_dependent=False,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self.nactions = env_spec.action_space.flat_dim

        # Notice the self.nactions-1 implicitly assumes conditional independence
        # TODO(cathywu) implement a more general version for general
        # factorized policies
        self._regressor = GaussianMLPRegressor(
            input_shape=((env_spec.observation_space.flat_dim + (
                self.nactions - 1) * action_dependent) * num_seq_inputs,),
            output_dim=1,
            name="vf",
            **regressor_args
        )

    @overrides
    def fit(self, paths, idx=None):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        if idx is not None:
            actions = np.concatenate([p["actions"][:, [x for x in range(
                self.nactions) if x != idx]] for p in paths])
            obs_actions = np.hstack((observations, actions))
            self._regressor.fit(obs_actions, returns.reshape((-1, 1)))
        else:
            self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path, idx=None):
        if idx is not None:
            minus_idx = [x for x in range(self.nactions) if x != idx]
            a = path["actions"][:, minus_idx]
            return self._regressor.predict(np.hstack((path["observations"],
                                                      a))).flatten()
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
