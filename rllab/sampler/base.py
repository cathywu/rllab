

import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
from rllab.misc import attr_utils
import rllab.misc.logger as logger


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
        self.action_dependent = attr_utils.is_action_dependent(self.algo.baseline)

    def process_baselines(self, baseline, path_baseline, path,
                          agent=None, nagents=None):
        if not attr_utils.is_action_dependent(baseline) and not attr_utils.is_shared_policy(
                self.algo):
            path_baselines = np.append(path_baseline, 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            # TODO(cathywu) what do the last 2 terms mean?
            # 1-step bellman error / TD error
            return path_baselines[:-1], deltas
        elif not attr_utils.is_shared_policy(self.algo):
            nactions = path["actions"].shape[-1]
            path_baselines = np.hstack([path_baseline, np.zeros((nactions, 1))])
            deltas = (np.tile(path["rewards"], [nactions, 1]) + \
                      self.algo.discount * path_baselines[:, 1:] - \
                      path_baselines[:, :-1]).T
            return path_baselines[:, :-1], deltas
        else:  # spatial discounting, no AD baseline
            path_baselines = np.append(path_baseline, 0)
            local_reward = path['env_infos']["local_reward"]
            positions = path['env_infos']["positions"]
            spatial_returns = special.spatial_discount(local_reward, agent,
                                                       positions,
                                                       gamma=self.algo.spatial_discount,
                                                       type=self.algo.spatial_discount_type)
            deltas = spatial_returns + self.algo.discount * path_baselines[
                1:] - path_baselines[:-1]
            return path_baselines[:-1], deltas

    @staticmethod
    def process_expected_variance(baseline, baselines, returns):
        if not attr_utils.is_action_dependent(baseline):
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
        nagents = len(paths[0]['observations'])

        if attr_utils.is_shared_policy(self.algo):
            returns = [[] for _ in range(nagents)]
            baselines = [[] for _ in range(nagents)]
        else:
            baselines = []
            returns = []

        if attr_utils.is_shared_policy(self.algo):
            all_path_baselines = [[self.algo.baseline[i].predict(path, agent=i)
                                        for path in paths] for i in
                                  range(nagents)]
            for idx, path in enumerate(paths):
                path_baseline, deltas = zip(*[self.process_baselines(
                    self.algo.baseline[i], all_path_baselines[i][idx],
                    path, agent=i, nagents=nagents) for i in range(nagents)])
                for i in range(nagents):
                    baselines[i].append(path_baseline[i])

                # Convention: [k, path_length]
                path["advantages"] = np.vstack([special.discount_cumsum(
                    deltas[i], self.algo.discount * self.algo.gae_lambda) for
                                      i in range(nagents)])

                local_reward = path['env_infos']["local_reward"]
                positions = path['env_infos']["positions"]
                for i in range(nagents):
                    spatial_returns = special.spatial_discount(local_reward, i,
                                                               positions,
                                                               gamma=self.algo.spatial_discount,
                                                               type=self.algo.spatial_discount_type)
                    path["returns-%s" % i] = special.discount_cumsum(
                        spatial_returns, self.algo.discount)
                    path["returns"] = special.discount_cumsum(path[
                        "rewards"], self.algo.discount)
                    returns[i].append(path["returns-%s" % i])

            ev = [self.process_expected_variance(self.algo.baseline[i],
                                                 baselines[i],
                                            np.concatenate(returns[i])) for i
                  in range(nagents)]
        else:
            if hasattr(self.algo.baseline, "predict_n"):
                all_path_baselines = self.algo.baseline.predict_n(paths)
            else:
                all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

            for idx, path in enumerate(paths):
                path_baseline, deltas = self.process_baselines(self.algo.baseline,
                    all_path_baselines[idx], path)
                baselines.append(path_baseline)

                path["advantages"] = special.discount_cumsum(
                    deltas, self.algo.discount * self.algo.gae_lambda)

                path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
                returns.append(path["returns"])
            ev = self.process_expected_variance(self.algo.baseline, baselines,
                                            np.concatenate(returns))

        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        if attr_utils.is_shared_policy(self.algo):
            # Convention: list of length k (agents) of tensors of size [path_length, agent_obs_dim]
            observations = [tensor_utils.concat_tensor_list([path["observations"][i] for path in paths]) for i in range(nagents)]
            # Convention: size [k agents, path length] for ease of flattening
            actions = [tensor_utils.concat_tensor_list([path["actions"][i] for path in paths]) for i in range(nagents)]
            # Convention: size [path length, k agents]
            returns = tensor_utils.stack_tensor_list([tensor_utils.concat_tensor_list([path[
                "returns-%s" % i] for path in paths]) for i in range(nagents)]).T
            # Convention: size [k agents, path length] for ease of flattening
            advantages = np.array(tensor_utils.concat_tensor_list([path["advantages"].T
                                                          for path in paths])).T
        else:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
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

        # Convention: [?, ...], where ? indicates the batch size
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

        logger.log("fitting baseline...")
        if attr_utils.is_shared_policy(self.algo):
            for i in range(nagents):
                self.algo.baseline[i].fit(paths, agent=i, returns="returns-%s" % i)
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
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
