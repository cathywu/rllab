import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.action_dependent_gaussian_mlp_baseline import \
    ActionDependentGaussianMLPBaseline
from rllab.baselines.action_dependent_linear_feature_baseline import \
    ActionDependentLinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.algos.trpo_action import TRPOAction as TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from nose2 import tools

baselines = [ActionDependentLinearFeatureBaseline,
             ActionDependentGaussianMLPBaseline]

@tools.params(*baselines)
def test_action_dependent_baseline(baseline_cls):
    env = TfEnv(normalize(GymEnv("Walker2d-v1", force_reset=True),
                      normalize_obs=False))
    try:
        policy = GaussianMLPPolicy(name="policy", env_spec=env.spec,
                               hidden_sizes=(6,))
    except ValueError:
        # FIXME(cathywu) It looks like nose2 does not start each run of this
        # parameterized test in an isolated manner; the tf variables are
        # already created by the first test, which causes problems for
        # subsequent runs. The current hack is to reuse variables,
        # which should be fine for a unit test, but shouldn't ever be used
        # for an actual experiment.
        tf.get_variable_scope().reuse_variables()
        policy = GaussianMLPPolicy(name="policy", env_spec=env.spec,
                                   hidden_sizes=(6,))
    baseline = baseline_cls(env_spec=env.spec)
    algo = TRPO(
        env=env, policy=policy, baseline=baseline,
        n_itr=1, batch_size=1000, max_path_length=100
    )
    algo.train()
