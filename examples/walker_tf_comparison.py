import tensorflow as tf

action_dependent = True

if action_dependent:
    from sandbox.rocky.tf.algos.trpo_action import TRPOAction as TRPO
else:
    from sandbox.rocky.tf.algos.trpo import TRPO

from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.action_dependent_linear_feature_baseline import ActionDependentLinearFeatureBaseline
from rllab.baselines.action_dependent_gaussian_mlp_baseline import ActionDependentGaussianMLPBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite

# algo = "VPG"
algo = "TRPO"

# exp_prefix = "Walker2d-comparison"
exp_prefix = "improve-action-baseline"

# Not needed for a local run
# stub(globals())

env = TfEnv(normalize(GymEnv("Walker2d-v1", force_reset=True),
                      normalize_obs=False))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    # hidden_sizes=(8,),
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.tanh,
)

if action_dependent:
    # baseline = ActionDependentLinearFeatureBaseline(env_spec=env.spec)
    baseline = ActionDependentGaussianMLPBaseline(env_spec=env.spec)
else:
    # baseline = LinearFeatureBaseline(env_spec=env.spec)
    baseline = GaussianMLPBaseline(env_spec=env.spec)
# TODO(cathywu) pass in arguments so that the baselines uses the same number
# of weights or something

# Parameters from https://github.com/shaneshixiang/rllabplusplus/blob/master/
# sandbox/rocky/tf/launchers/launcher_utils.py
# Parameter matching from https://github.com/shaneshixiang/rllabplusplus/blob/
# master/sandbox/rocky/tf/launchers/launcher_stub_utils.py
if algo == 'TRPO':
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=env.horizon,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        sample_backups=0,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        gae_lambda=0.97,

    )
elif algo == 'VPG':
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=env.horizon,
        n_itr=1000,
        discount=0.99,
        optimizer_args=dict(
            tf_optimizer_args=dict(
                learning_rate=0.001,
            )
        ),
        gae_lambda=0.97,
    )

run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
    exp_prefix=exp_prefix,
)
