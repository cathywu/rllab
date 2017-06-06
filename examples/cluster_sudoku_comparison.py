import datetime
import dateutil.tz
import sys
from random import shuffle
import os

import numpy as np
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

debug = True

exp_prefix = "cluster-sudoku-v2" if not debug \
    else "cluster-sudoku-debug"
mode = 'ec2' if not debug else 'local'  # 'local_docker', 'ec2', 'local'
n_itr = 2000 if not debug else 2
holdout_factor = 0.0

# Index among variants to start at
offset = 0

sizes = [2, 3]
arr = [0 for _ in range(len(sizes))]
for i, size in enumerate(sizes):
    fname = os.path.join('data', str(size), 'features.npy')
    arr[i] = np.load(fname)


def batch_size(size):
    if size == 4:
        return 1000
    elif size == 9:
        return 10000


def mat_to_mask(mat):
    size = mat.shape[0]
    mask = np.vstack(
        [[x for x in zip(*np.where(mat[i] == 1))] for i in range(size) if np.where(mat[i] == 1)[0].size > 0])
    values = [[i] * int(np.sum(mat[i])) for i in range(size) if int(np.sum(mat[i])) > 0]
    mask_values = [x for y in values for x in y]
    # print('BOARD', mat, mask, mask_values)
    return (size, mask.tolist(), mask_values, batch_size(size))


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
        return [16]  # [True, False]

    @variant
    def board(self):
        return [mat_to_mask(arr[1][i]) for i in range(2)]
        # return [mat_to_mask(arr[0][i]) for i in range(20)] + \
        # [mat_to_mask(arr[1][i]) for i in range(20)]

        # return [(
        #   4, [[0, 1], [1, 3], [2, 0], [3, 2]], [2, 3, 1, 1], 1000),
        # ]

        # For multi sudoku env
        # configs = [
        #     [mat_to_mask(arr[0][i]) for i in range(16)],
        #     [mat_to_mask(arr[0][i]) for i in range(16, 16 + 16)],
        #     [mat_to_mask(arr[0][i]) for i in range(16 * 2, 16 * 2 + 16)],
        # ]
        # return [(4, [c[1] for c in config], [c[2] for c in config], 8000) for
        #         config in configs]

        # [mat_to_mask(arr[1][i]) for i in range(20)]
        # return [
        #     (4, [[0, 1], [1, 3], [2, 0], [3, 2]], [2, 3, 1, 1], 1000),
        # ]
        # (
        # 4, np.array([[0, 2], [0, 3], [3, 0], [3, 1]]), np.array([2, 1, 3, 2]),
        # 1000), (
        # 9, np.array([[0, 2], [0, 3], [3, 0], [3, 1]]), np.array([2, 1, 3, 2]),
        # 1000), ]

    @variant
    def exit_when_done(self):
        return [True]  # [True, False]

    @variant
    def collisions(self):
        return [True]  # [True]  # [False, True]

    @variant
    def max_path_length(self):
        return [50]  # [10]  # [50, 200, 1000]

    @variant
    def step_size(self):
        return [0.01]  # , 0.05, 0.1]

    @variant
    def baseline_mix_fraction(self):
        return [1.0]  # , 0.1]  # [0.2, 0.1, 1.0]

    @variant
    def baseline_include_time(self):
        return [True]  # , False]

    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]  # 1, 21, 31, 41]

    @variant
    def gae_lambda(self):
        return [1.0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]

    @variant
    def env(self):
        return [
            # "OneStepNoStateEnv",
            # "NoStateEnv",
            # "MultiagentPointEnv",
            # "MultigoalEnv",
            # "MultiSudokuEnv",
            "SudokuEnv",
            # "MultiactionPointEnv",
        ]


def gen_run_task(baseline_cls):
    def run_task(vv):
        if vv['env'] == "MultiagentPointEnv":
            from rllab.envs.multiagent_point_env import \
                MultiagentPointEnv as MEnv
        elif vv['env'] == "MultiactionPointEnv":
            from rllab.envs.multiaction_point_env import \
                MultiactionPointEnv as MEnv
        elif vv['env'] == "NoStateEnv":
            from rllab.envs.no_state_env import NoStateEnv as MEnv
        elif vv['env'] == "OneStepNoStateEnv":
            from rllab.envs.one_step_no_state_env import \
                OneStepNoStateEnv as MEnv
        elif vv['env'] == "MultigoalEnv":
            from rllab.envs.multigoal_env import MultigoalEnv as MEnv
        elif vv['env'] == "SudokuEnv":
            from rllab.envs.sudoku_env import SudokuEnv as MEnv
        elif vv['env'] == "MultiSudokuEnv":
            from rllab.envs.multi_sudoku_env import MultiSudokuEnv as MEnv
        else:
            raise NotImplementedError
        # running average normalization
        env = TfEnv(NormalizedEnv(NormalizeObs(
            MEnv(horizon=vv['max_path_length'], k=vv['k'], d=vv['board'][0],
                 mask=vv['board'][1], mask_values=vv['board'][2]), clip=5)))

        # exponential weighting normalization
        # env = TfEnv(normalize(MultiagentPointEnv(d=1, k=6),
        #                       normalize_obs=True))

        policy = AutoMLPPolicy(
            env_spec=env.spec,
            name="policy",
            hidden_sizes=(),
            # hidden_sizes=(100, 50, 25),
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline_args = {'env_spec': env.spec,
            'mix_fraction': vv["baseline_mix_fraction"],
            'include_time': vv["baseline_include_time"],
            'regressor_args': {'holdout_factor': holdout_factor, }}
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
            batch_size=vv['board'][3],
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
