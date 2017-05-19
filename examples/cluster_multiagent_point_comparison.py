import datetime
import dateutil.tz
import sys
from random import shuffle

import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.action_dependent_linear_feature_baseline import \
    ActionDependentLinearFeatureBaseline
from rllab.baselines.action_dependent_gaussian_mlp_baseline import \
    ActionDependentGaussianMLPBaseline
from sandbox.rocky.tf.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalize_obs import NormalizeObs
# from rllab.envs.normalized_env import normalize
from rllab.envs.normalized_env import NormalizedEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.auto_mlp_policy import AutoMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from rllab.baselines import util

from rllab.misc.instrument import VariantGenerator, variant
from rllab import config
from rllab import config_personal

debug = False

exp_prefix = "cluster-multiagent-v25" if not debug \
    else "cluster-multiagent-debug"
mode = 'ec2' if not debug else 'local'  # 'local_docker', 'ec2', 'local'
n_itr = 2000 if not debug else 2
holdout_factor = 0.0

# Index among variants to start at
offset = 0


class VG(VariantGenerator):
    @variant
    def baseline(self):
        return [
            # "LinearFeatureBaseline",
            "ActionDependentLinearFeatureBaseline",
            # "ZeroBaseline",
            # "ActionDependentGaussianMLPBaseline",
            # "GaussianMLPBaseline",
        ]

    @variant
    def k(self):
        return [6, 50, 200]  # [6, 50, 200]  # , 500]  # [6, 50, 200, 500, 1000]

    @variant
    def d(self):
        return [2]  # [1, 2] # [1, 2, 10]

    @variant
    def exit_when_done(self):
        return [True]  # [True, False]

    @variant
    def collisions(self):
        return [True]  # [True]  # [False, True]

    @variant
    def batch_size(self):
        return [
            100 / (1.0-holdout_factor),
            # 500 / (1.0-holdout_factor),
            1000 / (1.0-holdout_factor),
            # 5000 / (1.0-holdout_factor),
            # 10000 / (1.0-holdout_factor),
            # 25000,
        ]

    @variant
    def corridor(self):
        return [0.4]  # [0.2, 0.3, 0.4]

    @variant
    def done_epsilon(self):
        return [0.01]  # , 0.05]  # [0.05, 0.005]

    @variant
    def done_reward(self):
        return [1]  # [1000, 100]  # , 200, 100, 50]  # 10

    @variant
    def goal_weight(self):
        return [0]  # [0, 1]

    @variant
    def collision_epsilon(self):
        return [0.05]  # [0.5, 0.005]  # 0.1

    @variant
    def collision_penalty(self):
        return [10]  # [1000, 100]  # , 200, 100, 50]  # 10

    @variant
    def ignore_intra_collisions(self):
        return [False]  # [True]  # , False]

    @variant
    def repeat_action(self):
        return [1]  # [5]

    @variant
    def max_path_length(self):
        return [1, 2, 3, 4, 5]  # [50]  # [10]  # [50, 200, 1000]

    @variant
    def step_size(self):
        return [0.01]  # , 0.05, 0.1]

    @variant
    def baseline_mix_fraction(self):
        return [1.0]  #, 0.1]  # [0.2, 0.1, 1.0]

    @variant
    def baseline_include_time(self):
        return [True]  # , False]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]  # 1, 21, 31, 41]

    @variant
    def gae_lambda(self):
        return [0.3, 0.7, 0.97]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]

    @variant
    def env(self):
        return [
            # "OneStepNoStateEnv",
            # "NoStateEnv",
            "MultiagentPointEnv",
            # "MultigoalEnv",
            # "MultiactionPointEnv",
        ]


def gen_run_task(baseline_cls):
    def run_task(vv):
        if vv['env'] == "MultiagentPointEnv":
            from rllab.envs.multiagent_point_env import MultiagentPointEnv as MEnv
        elif vv['env'] == "MultiactionPointEnv":
            from rllab.envs.multiaction_point_env import MultiactionPointEnv as MEnv
        elif vv['env'] == "NoStateEnv":
            from rllab.envs.no_state_env import NoStateEnv as MEnv
        elif vv['env'] == "OneStepNoStateEnv":
            from rllab.envs.one_step_no_state_env import OneStepNoStateEnv as MEnv
        elif vv['env'] == "MultigoalEnv":
            from rllab.envs.multigoal_env import MultigoalEnv as MEnv
        # running average normalization
        env = TfEnv(NormalizedEnv(NormalizeObs(
            MEnv(d=vv['d'], k=vv['k'], horizon=vv['max_path_length'],
                 collisions=vv['collisions'],
                 exit_when_done=vv['exit_when_done'],
                 done_epsilon=vv['done_epsilon'],
                 done_reward=vv['done_reward'],
                 collision_epsilon=vv['collision_epsilon'],
                 collision_penalty=vv['collision_penalty'],
                 corridor=vv['corridor'],
                 repeat=vv['repeat_action'],
                 ignore_intra_collisions=vv['ignore_intra_collisions'],
                 goal_weight=vv['goal_weight']), clip=5)))

        # exponential weighting normalization
        # env = TfEnv(normalize(MultiagentPointEnv(d=1, k=6),
        #                       normalize_obs=True))

        policy = AutoMLPPolicy(
            env_spec=env.spec,
            name="policy",
            hidden_sizes=(200, 200),
            # hidden_sizes=(100, 50, 25),
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline_args = {
            'env_spec': env.spec,
            'mix_fraction': vv["baseline_mix_fraction"],
            'include_time': vv["baseline_include_time"],
            'regressor_args': {
                'holdout_factor': holdout_factor,
            }
        }
        if baseline_cls == "ActionDependentGaussianMLPBaseline":
            baseline = ActionDependentGaussianMLPBaseline(**baseline_args)
        elif baseline_cls == "ActionDependentLinearFeatureBaseline":
            baseline = ActionDependentLinearFeatureBaseline(**baseline_args)
        elif baseline_cls == "GaussianMLPBaseline":
            baseline = GaussianMLPBaseline(**baseline_args)
        elif baseline_cls == "LinearFeatureBaseline":
            baseline = LinearFeatureBaseline(**baseline_args)
        elif baseline_cls == "ZeroBaseline":
            baseline = ZeroBaseline(**baseline_args)
        action_dependent = util.is_action_dependent(baseline)
        if action_dependent:
            from sandbox.rocky.tf.algos.trpo_action import TRPOAction as TRPO
        else:
            from sandbox.rocky.tf.algos.trpo import TRPO

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=vv['batch_size'],
            max_path_length=vv['max_path_length'],
            n_itr=n_itr,  # 1000
            discount=0.995,
            step_size=vv["step_size"],
            sample_backups=0,
            # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
            gae_lambda=vv['gae_lambda'],
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
            center_adv=False,  # This disables whitening of advantages
            # extra_baselines=[LinearFeatureBaseline(**baseline_args),
            #                  ZeroBaseline(**baseline_args)],
        )
        algo.train()

    return run_task


variants = VG().variants()

SERVICE_LIMIT = 400
# AWS_REGIONS = [x for x in config_personal.ALL_REGION_AWS_KEY_NAMES.keys()]
AWS_REGIONS = ['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2']
shuffle(AWS_REGIONS)
print("AWS REGIONS order", AWS_REGIONS)

for i, v in enumerate(variants):

    if i < offset:
        continue

    if mode == "ec2" and i - offset >= len(AWS_REGIONS) * SERVICE_LIMIT:
        sys.exit()

    print("Issuing variant %s: %s" % (i, v))

    if mode == "ec2":
        config.AWS_REGION_NAME = AWS_REGIONS[(i - offset) % len(AWS_REGIONS)]
        config.AWS_KEY_NAME = config_personal.ALL_REGION_AWS_KEY_NAMES[
            config.AWS_REGION_NAME]
        config.AWS_IMAGE_ID = config_personal.ALL_REGION_AWS_IMAGE_IDS[
            config.AWS_REGION_NAME]
        config.AWS_SECURITY_GROUP_IDS = \
            config_personal.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                config.AWS_REGION_NAME]

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    run_experiment_lite(
        gen_run_task(v["baseline"]),
        exp_prefix=exp_prefix,
        # exp_name="%s_%s_%04d" % (exp_prefix, timestamp, i),
        # Number of parallel workers for sampling
        n_parallel=1,  # not used for tf implementation
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        # mode="local",
        # mode="ec2",
        mode=mode,
        # mode="local_docker",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    if debug:
        sys.exit()
