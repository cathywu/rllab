

import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
from rllab.baselines import util as bs_util
import rllab.misc.logger as logger
from rllab.envs.sudoku_env import SudokuEnv


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class BaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.nactions = self.algo.nactions
        self.action_dependent = bs_util.is_action_dependent(self.algo.baseline)

    def process_baselines(self, baseline, path_baseline, path):
        if not bs_util.is_action_dependent(baseline):
            path_baselines = np.append(path_baseline, 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            # TODO(cathywu) what do the last 2 terms mean?
            # 1-step bellman error / TD error
            return path_baselines[:-1], deltas
        else:
            path_baselines = np.hstack([path_baseline, np.zeros((self.nactions, 1))])
            deltas = (np.tile(path["rewards"], [self.nactions, 1]) + \
                      self.algo.discount * path_baselines[:, 1:] - \
                      path_baselines[:, :-1]).T
            return path_baselines[:, :-1], deltas

    @staticmethod
    def process_expected_variance(baseline, baselines, returns):
        if not bs_util.is_action_dependent(baseline):
            ev = special.explained_variance_1d(
                np.concatenate(baselines), returns)
        else:
            nactions = baselines[0].shape[0]
            ev = [0 for _ in range(nactions)]
            for k in range(nactions):
                ev[k] = special.explained_variance_1d(
                    np.concatenate([b[k, :] for b in baselines]), returns)
        return ev

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            if self.action_dependent and isinstance(self.algo.env, SudokuEnv):
                # Sudoku manual computation of AD baselines
                all_path_baselines = []
                d = self.algo.env.d
                for path in paths:
                    A = path['actions'].reshape((-1, d))
                    board = np.where(A == 1)[1].reshape((d, d))
                    # v0 = -SudokuEnv.violations(board)
                    baseline = []
                    for i in range(d):
                        for j in range(d):
                            B = np.copy(board)
                            vs = []
                            for k in range(d):
                                B[i, j] = k
                                vs.append(self.algo.env.score(B))
                            # reward for average action
                            baseline.append(np.mean(vs))
                    all_path_baselines.append(np.expand_dims(np.array(baseline),
                                                            axis=-1))
            else:
                all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baseline, deltas = self.process_baselines(self.algo.baseline,
                all_path_baselines[idx], path)
            baselines.append(path_baseline)

            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            # import ipdb
            # ipdb.set_trace()

            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            returns.append(path["returns"])

        ev = self.process_expected_variance(self.algo.baseline, baselines,
                                            np.concatenate(returns))

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            advantages = util.center_advantages(advantages)

        if self.algo.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )

        if self.algo.extra_baselines is not None:
            n_extra_baselines = len(self.algo.extra_baselines)
            extra_ap_baselines = [0 for _ in range(n_extra_baselines)]
            extra_ev = [0 for _ in range(n_extra_baselines)]

            for i, b in enumerate(self.algo.extra_baselines):
                if hasattr(b, "predict_n"):
                    extra_ap_baselines[i] = b.predict_n(paths)
                else:
                    extra_ap_baselines[i] = [b.predict(path) for path in paths]

                extra_baselines = []
                for idx, path in enumerate(paths):
                    path_baseline, deltas = self.process_baselines(b,
                                                                   extra_ap_baselines[i][idx], path)
                    extra_baselines.append(path_baseline)

                extra_ev[i] = self.process_expected_variance(
                    b, extra_baselines, returns)


        logger.log("fitting baseline...")
        if self.action_dependent and isinstance(self.algo.env, SudokuEnv):
            pass
        else:
            if hasattr(self.algo.baseline, 'fit_with_samples'):
                self.algo.baseline.fit_with_samples(paths, samples_data)
            else:
                self.algo.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        if isinstance(ev, list):
            for k in range(len(ev)):
                logger.record_tabular('ExplainedVariance[%s]' % k, ev[k])
            ev_mean = np.mean(ev)
            logger.record_tabular('ExplainedVariance', ev_mean)
        else:
            logger.record_tabular('ExplainedVariance', ev)
            ev_mean = ev
        if self.algo.extra_baselines is not None:
            for i, b in enumerate(self.algo.extra_baselines):
                if isinstance(ev, list):
                    logger.record_tabular('EV[%s]' % type(b).__name__,
                                          np.mean(extra_ev[i]))
                    logger.record_tabular('dEV[%s]' % type(b).__name__,
                                          np.mean(extra_ev[i]) - ev_mean)
                else:
                    logger.record_tabular('EV[%s]' % type(b).__name__,
                                          extra_ev[i])
                    logger.record_tabular('dEV[%s]' % type(b).__name__,
                                          extra_ev[i] - ev_mean)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
