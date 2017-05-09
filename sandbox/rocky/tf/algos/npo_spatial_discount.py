import numpy as np
import tensorflow as tf

from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian


class NPOSpatialDiscount(BatchPolopt):
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
        self.spatial_discounting = True
        super(NPOSpatialDiscount, self).__init__(**kwargs)


    @overrides
    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
            size=self.env.per_agent_obsdim,
        )
        # Convention: shape [?, 1]
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
            size=self.env.per_agent_actiondim,
        )
        # Convention: shape [?, 1]
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        # Convention: shape [?, nactions]
        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        mean_kl = tf.reduce_mean(kl)
        loss_vec = lr * advantage_var
        surr_loss = - tf.reduce_mean(loss_vec)

        input_list = ext.flatten_list([
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        # TODO(cathywu) remove, for debugging
        self.lr = lr
        self.advantage_var = advantage_var
        self.input_list = input_list
        self.action_var = action_var
        self.mean_kl = mean_kl
        self.loss_vec = loss_vec
        self.surr_loss = surr_loss

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.flatten_list(ext.extract(
            samples_data,
            "observations_stack", "actions_stack", "advantages_flatten_vec"
        )))

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        # Convention: flatten column-wise, so that samples from the first agent are all together anagent-wise
        N = agent_infos['mean'].shape[0]
        reshape = lambda x: np.hstack(x).reshape([N, self.env.nagents, -1])
        dist_info_list = [np.vstack(reshape(agent_infos[k])) for k in
                      self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        # TODO(cathywu) remove, for debugging
        temp = tensor_utils.compile_function(self.input_list,
                                             self.advantage_var)
        adv = temp(*all_input_values)
        # print("CATHYWU adv", temp(*all_input_values))
        assert(len(adv.shape) == 1)

        temp = tensor_utils.compile_function(self.input_list,
                                             self.lr)
        lr = temp(*all_input_values)
        # print("cathywu lrs", temp(*all_input_values))
        assert(len(lr.shape) == 1)

        temp = tensor_utils.compile_function(self.input_list,
                                             self.surr_loss)
        surr_loss = temp(*all_input_values)
        print("CATHYWU surr_loss", surr_loss)

        temp = tensor_utils.compile_function(self.input_list,
                                             self.loss_vec)
        loss_vec = temp(*all_input_values)
        # print("CATHYWU loss_vec", loss_vec)
        partial_loss_manual = np.sum(lr * adv)
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
            env=self.env,
        )
