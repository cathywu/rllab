from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.distributions.product_distribution import ProductDistribution
import tensorflow as tf
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.product import Product


class NPOAction(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        # TODO(cathywu) put this elsewhere
        self.action_baseline = True
        super(NPOAction, self).__init__(**kwargs)


    @overrides
    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        # Convention: shape [?, 1]
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        # Convention: shape [?, 1]
        advantage_vars = [tensor_utils.new_tensor(
            'advantage',
            ndim=1,
            dtype=tf.float32,
            size=1,
        ) for _ in range(self.nactions)]
        dist = self.policy.distribution

        # Convention: shape [?, nactions]
        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1) + list(shape),
                              name='old_%s' % k) for k, shape in
            dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        # FIXME(cathywu) hack to handle sudoku env and multiagent point envs
        if isinstance(self.env.action_space, Box):
            dist1 = DiagonalGaussian(1)

        lrs = [0 for _ in range(self.nactions)]
        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        for k in range(self.nactions):
            # FIXME(cathywu) hack to handle sudoku env and multiagent point envs
            if isinstance(self.env.action_space, Product):
                lrs[k] = tf.expand_dims(dist.likelihood_ratio_sym(action_var,
                                                                  old_dist_info_vars,
                                                                  dist_info_vars, idx=k), axis=1)
            elif isinstance(self.env.action_space, Box):
                # Convention: shape [?, 1]
                lrs[k] = tf.expand_dims(dist1.likelihood_ratio_sym(action_var,
                                                                  old_dist_info_vars,
                                                                  dist_info_vars, idx=k), axis=1)

        mean_kl = tf.reduce_mean(kl)
        # Sum of product between adv and ratio of likelihoods
        loss_vec = tf.add_n([lrs[k] * advantage_vars[k]
                                               for k in range(self.nactions)])
        surr_loss = - tf.reduce_mean(loss_vec)

        input_list = ext.flatten_list([
                         obs_var,
                         action_var,
                         advantage_vars,
                     ] + state_info_vars_list + old_dist_info_vars_list)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        # TODO(cathywu) remove, for debugging
        self.lrs = lrs
        self.advantage_vars = advantage_vars
        self.input_list = input_list
        self.action_var = action_var
        self.mean_kl = mean_kl
        self.loss_vec = loss_vec
        self.surr_loss = surr_loss

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        # TODO(cathywu) need to compute nactions x advantages
        all_input_values = tuple(ext.flatten_list(ext.extract(
            samples_data,
            "observations", "actions", "advantages_single"
        )))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in
                          self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        # TODO(cathywu) remove, for debugging
        temp = tensor_utils.compile_function(self.input_list,
                                             self.advantage_vars)
        adv = temp(*all_input_values)
        # print("CATHYWU adv", temp(*all_input_values))
        assert(adv[0].shape[-1] == 1)

        temp = tensor_utils.compile_function(self.input_list,
                                             self.lrs)
        lrs = temp(*all_input_values)
        # print("cathywu lrs", temp(*all_input_values))
        assert(lrs[0].shape[-1] == 1)

        temp = tensor_utils.compile_function(self.input_list,
                                             self.surr_loss)
        surr_loss = temp(*all_input_values)
        print("CATHYWU surr_loss", surr_loss)

        temp = tensor_utils.compile_function(self.input_list,
                                             self.loss_vec)
        loss_vec = temp(*all_input_values)
        # print("CATHYWU loss_vec", loss_vec)
        import numpy as np
        partial_loss_manual = np.sum([lrs[k] * adv[k] for k in range(
            self.nactions)], axis=0)
        # print("cathywu loss terms", partial_loss_manual)
        print("cathywu loss", -np.mean(partial_loss_manual))
        print("cathywu mean loss_vec", -np.mean(loss_vec))
        # FIXME(cathywu) ASK ROCKY manually computed loss and surr_loss disagree
        # import ipdb
        # ipdb.set_trace()

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.sampler.vec_env.envs[0], # self.env,
        )
