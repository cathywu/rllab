import time

import tensorflow as tf

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.misc import ext
from rllab.misc.ext import sliced_fun


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            extra_baselines=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.extra_baselines = extra_baselines
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.nactions = self.env.action_space.flat_dim
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, max_samples=None, log=True):
        return self.sampler.obtain_samples(itr, max_samples=max_samples,
                                           log=log)

    def process_samples(self, itr, paths, update_baseline=True, log=True):
        return self.sampler.process_samples(itr, paths, log=log,
                                            update_baseline=update_baseline)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    # TODO(cathywu) this should be a parameter somewhere
                    n_independent_batches = 5
                    gradient_estimates = []
                    for batch in range(n_independent_batches):
                        update_baseline = True if batch == \
                            n_independent_batches - 1 else False
                        log = True if batch == n_independent_batches - 1 else False
                        max_samples = self.batch_size * 100 if batch == 0 and\
                                                               n_independent_batches > 1 else None

                        logger.log("Obtaining samples...")
                        paths = self.obtain_samples(itr,
                                                    log=log,
                                                    max_samples=max_samples)
                        logger.log("Processing samples...")
                        if self.extra_baselines is not None:
                            samples_datas = self.process_samples(itr, paths,
                                                                 log=log,
                                                                update_baseline=update_baseline)
                            samples_data = samples_datas[0]
                        else:
                            samples_data = self.process_samples(itr, paths, log=log,
                                                            update_baseline=update_baseline)
                        # TODO(cathywu) compute gradient estimate for each
                        # sample data; Is the gradient ever explicitly computed?
                        all_input_values = tuple(ext.flatten_list(ext.extract(
                            samples_data,
                            "observations", "actions", "advantages_single"
                        )))
                        agent_infos = samples_data["agent_infos"]
                        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
                        dist_info_list = [agent_infos[k] for k in
                                          self.policy.distribution.dist_info_keys]
                        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
                        flat_g = sliced_fun(self.optimizer._opt_fun["f_grad"],
                                            self._num_slices)(
                            all_input_values, None)
                        gradient_estimates.append(0)

                        if batch != n_independent_batches - 1:
                            continue
                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(paths)
                        logger.log("Optimizing policy...")
                        self.optimize_policy(itr, samples_data)
                        logger.log("Saving snapshot...")
                        params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                        if self.store_paths:
                            params["paths"] = samples_data["paths"]
                        logger.save_itr_params(itr, params)
                        logger.log("Saved")
                        logger.record_tabular('Time', time.time() - start_time)
                        logger.record_tabular('ItrTime', time.time() - itr_start_time)
                        logger.dump_tabular(with_prefix=False)
                        if self.plot:
                            self.update_plot()
                            if self.pause_for_plot:
                                input("Plotting evaluation run: Press Enter to "
                                      "continue...")
                    # TODO(cathywu) log empirical estimate of variance of
                    # policy gradient per parameter

        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
