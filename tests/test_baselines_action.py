import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.action_dependent_gaussian_mlp_baseline import \
    ActionDependentGaussianMLPBaseline
from rllab.baselines.action_dependent_linear_feature_baseline import \
    ActionDependentLinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalize_obs import NormalizeObs
from sandbox.rocky.tf.algos.trpo_action import TRPOAction as TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from nose2 import tools

baselines = [ActionDependentLinearFeatureBaseline,
             ActionDependentGaussianMLPBaseline]

@tools.params(*baselines)
def test_action_dependent_baseline(baseline_cls):
    env = TfEnv(NormalizeObs(GymEnv("Walker2d-v1", force_reset=True),
                      clip=5))
    policy = GaussianMLPPolicy(name="policy-%s" % baseline_cls.__name__,
                               env_spec=env.spec,
                               hidden_sizes=(6,))
    baseline = baseline_cls(env_spec=env.spec)
    algo = TRPO(
        env=env, policy=policy, baseline=baseline,
        n_itr=1, batch_size=1000, max_path_length=100
    )
    algo.train()
