import os

from rllab.envs.normalize_obs import NormalizeObs
from rllab.envs.gym_env import GymEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from examples.multiagent_point_env import MultiagentPointEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from nose2 import tools


baselines = [ZeroBaseline, LinearFeatureBaseline, GaussianMLPBaseline]
max_path_length = 1000


@tools.params(*baselines)
def test_baseline(baseline_cls):
    # env = TfEnv(NormalizeObs(GymEnv("Walker2d-v1", force_reset=True,
    #                              record_video=False, record_log=False),
    #                       clip=5))
    env = TfEnv(NormalizeObs(MultiagentPointEnv(d=1, k=6,
                                                horizon=max_path_length),
                             clip=5))
    policy = GaussianMLPPolicy(name="policy-%s" % baseline_cls.__name__,
                               env_spec=env.spec,
                               hidden_sizes=(6,))
    baseline = baseline_cls(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=max_path_length,
        n_itr=2,
        discount=0.99,
        step_size=0.01,
        sample_backups=0,
        gae_lambda=0.97,
    )
    algo.train()
