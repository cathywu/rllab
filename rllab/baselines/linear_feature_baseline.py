import numpy as np

from sandbox.rocky.tf.baselines.base import Baseline
from sandbox.rocky.tf.misc import space_utils
from sandbox.rocky.tf.regressors.moving_target_regressor import MovingTargetRegressor
from sandbox.rocky.tf.regressors.linear_regressor import LinearRegressor


class LinearFeatureBaseline(Baseline):
    def __init__(
            self,
            env_spec,
            reg_coeff=1e-5,
            mix_fraction=1.,
            include_time=True,
            action_dependent=False,
            spatial_discounting=False,
            **kwargs
    ):
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.mix_fraction = mix_fraction
        self.include_time = include_time
        self.nactions = env_spec.action_space.flat_dim
        self.action_dependent = action_dependent
        self.spatial_discounting = spatial_discounting
        self.regressor = MovingTargetRegressor(
            LinearRegressor(
                input_size=self.feature_size,
                output_size=1,
                reg_coeff=reg_coeff,
            ),
            mix_fraction=mix_fraction
        )

    def get_features(self, path, agent=None, idx=None):
        if agent is None:
            obs = path["observations"]
        else:
            obs = path["observations"][agent]
        o = np.clip(obs, -10, 10)
        l = len(path["rewards"])
        feats = [o, o ** 2]
        if idx is not None:
            minus_idx = [x for x in range(self.nactions) if x != idx]
            a = path["actions"][:, minus_idx]
            feats.extend([a, a ** 2])
        if self.include_time:
            al = np.arange(l).reshape(-1, 1) / 100.0
            feats.extend([al, al ** 2, al ** 3])
        return np.concatenate(feats, axis=-1)

    @property
    def feature_size(self):
        if self.spatial_discounting:
            # TODO(cathywu) Refactor this
            obs_dim = int(space_utils.space_to_flat_dim(self.observation_space)
                          / self.observation_space.shape[0])
        else:
            obs_dim = space_utils.space_to_flat_dim(self.observation_space)
        fsize = obs_dim * 2  # Is this 2 from obs + last_obs or o, o ** 2?
        if self.include_time:
            fsize += 3
        if self.action_dependent:
            fsize += (self.nactions - 1) * 2
        return fsize

    def fit(self, paths, agent=None, idx=None, returns="returns"):
        # don't regress against the last value in each path
        featmat = np.concatenate([self.get_features(path, agent=agent, idx=idx) for path
                                  in paths])
        returns = np.concatenate([path[returns] for path in paths])
        self.regressor.fit(featmat, returns.reshape((-1, 1)))

    def predict(self, path, agent=None, idx=None):
        feats = self.get_features(path, agent=agent, idx=idx)
        return self.regressor.predict_n(feats)[..., 0]
