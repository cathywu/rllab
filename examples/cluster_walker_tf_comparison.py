import sys

import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.action_dependent_linear_feature_baseline import ActionDependentLinearFeatureBaseline
from rllab.baselines.action_dependent_gaussian_mlp_baseline import ActionDependentGaussianMLPBaseline

from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.rocky.tf.algos.trpo import TRPO
# from sandbox.rocky.tf.algos.trpo_action import TRPOAction as TRPO
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.gym_env import GymEnv

from rllab.misc.instrument import VariantGenerator, variant

exp_prefix = "cluster_Walker2d_comparison"

class VG(VariantGenerator):

    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]


def gen_run_task(baseline_cls):

    def run_task(vv):
        env = TfEnv(normalize(GymEnv("Walker2d-v1", force_reset=True,
                                     record_video=False, record_log=False),
                          normalize_obs=False))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            name="policy",
            hidden_sizes=(100, 50, 25),
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = baseline_cls(env_spec=env.spec)
        action_dependent = True if (hasattr(baseline,
                                            "action_dependent") and baseline.action_dependent is True) else False
        if action_dependent:
            from sandbox.rocky.tf.algos.trpo_action import TRPOAction as TRPO
        else:
            from sandbox.rocky.tf.algos.trpo import TRPO

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=5000,
            max_path_length=env.horizon,
            n_itr=800, # 1000
            discount=0.99,
            step_size=vv["step_size"],
            sample_backups=0,
            # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
            gae_lambda=0.97,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
        algo.train()

    return run_task


variants = VG().variants()

baselines = [
             ActionDependentGaussianMLPBaseline,
             GaussianMLPBaseline,
             LinearFeatureBaseline,
             ActionDependentLinearFeatureBaseline,
            ]

for v in variants:

    for baseline_cls in baselines:

        run_experiment_lite(
            gen_run_task(baseline_cls),
            exp_prefix=exp_prefix,
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=v["seed"],
            # mode="local",
            mode="ec2",
            # mode="local_docker",
            variant=v,
            # plot=True,
            # terminate_machine=False,
        )
        # sys.exit()
