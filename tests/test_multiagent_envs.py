import os

from rllab.envs.normalize_obs import NormalizeObs
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.multiagent_point_env import MultiagentPointEnv
from rllab.envs.multiaction_point_env import MultiactionPointEnv
from rllab.envs.no_state_env import NoStateEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from nose2 import tools

envs = [
    (MultiagentPointEnv, 1, 6, True),
    (MultiagentPointEnv, 1, 6, False),
    (MultiagentPointEnv, 2, 6, True),
    (MultiagentPointEnv, 2, 6, False),
    (MultiactionPointEnv, 1, 6, False),
    (MultiactionPointEnv, 2, 6, False),
    (NoStateEnv, 1, 6, False),
    (NoStateEnv, 2, 6, False),
]
max_path_length = 30


@tools.params(*envs)
def test_multiagent_envs(env_cls, d, k, collisions):
    env = TfEnv(NormalizeObs(env_cls(d=d, k=k, collisions=collisions,
                                     horizon=max_path_length),
                             clip=5))
    tag = "%s-%s-%s-%s" % (env_cls.__name__, d, k, collisions)
    policy = GaussianMLPPolicy(name="policy-%s" % tag, env_spec=env.spec,
                               hidden_sizes=(6,))
    baseline = GaussianMLPBaseline(scope="vf-%s" % tag, env_spec=env.spec)
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
