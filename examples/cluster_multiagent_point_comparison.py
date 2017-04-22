import sys

import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.action_dependent_linear_feature_baseline import ActionDependentLinearFeatureBaseline
from rllab.baselines.action_dependent_gaussian_mlp_baseline import ActionDependentGaussianMLPBaseline

from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from examples.multiagent_point_env import MultiagentPointEnv

from rllab.misc.instrument import VariantGenerator, variant

exp_prefix = "cluster_multiagent_point_v2"

class VG(VariantGenerator):

    @variant
    def step_size(self):
        return [0.01, 0.05, 0.1]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]

    @variant
    def baseline(self):
        return [
            "ActionDependentGaussianMLPBaseline",
            "GaussianMLPBaseline",
             "LinearFeatureBaseline",
             "ActionDependentLinearFeatureBaseline",
        ]


def gen_run_task(baseline_cls):

    def run_task(vv):
        env = TfEnv(normalize(MultiagentPointEnv(d=1, k=6),
                          normalize_obs=False))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            name="policy",
            hidden_sizes=(100, 50, 25),
            hidden_nonlinearity=tf.nn.tanh,
        )

        if baseline_cls == "ActionDependentGaussianMLPBaseline":
            baseline = ActionDependentGaussianMLPBaseline(env_spec=env.spec)
        elif baseline_cls == "ActionDependentLinearFeatureBaseline":
            baseline = ActionDependentLinearFeatureBaseline(env_spec=env.spec)
        elif baseline_cls == "GaussianMLPBaseline":
            baseline = GaussianMLPBaseline(env_spec=env.spec)
        elif baseline_cls == "LinearFeatureBaseline":
            baseline = LinearFeatureBaseline(env_spec=env.spec)
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
            max_path_length=1000,
            n_itr=500, # 1000
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

for v in variants:

    run_experiment_lite(
        gen_run_task(v["baseline"]),
        exp_prefix=exp_prefix,
        # Number of parallel workers for sampling
        n_parallel=1,  # not used for tf implementation
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        # mode="local",
        # mode="ec2",
        mode="local_docker",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    # sys.exit()
