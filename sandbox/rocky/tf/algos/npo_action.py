


from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
import tensorflow as tf


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
        action_vars = [self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
            size=False,
        ) for _ in range(self.nactions)]
        advantage_vars = [tensor_utils.new_tensor(
            'advantage',
            ndim=1,
            dtype=tf.float32,
        ) for _ in range(self.nactions)]
        dist = self.policy.distribution

        print("CATHYWU", dist.dist_info_specs, self.policy.state_info_specs)

        dist1 = DiagonalGaussian(1)
        old_dist_info_varss = [{
            k: tf.placeholder(tf.float32, shape=[None] * (1) + list(shape),
                              name='old_%s_%s' % (k,j))
            for k, shape in dist1.dist_info_specs
            } for j in range(self.nactions)]
        print("CATHYWU old dist info vars", old_dist_info_varss)
        old_dist_info_vars_list = [old_dist_info_vars[k] for
                                   old_dist_info_vars in old_dist_info_varss
                                   for k in dist1.dist_info_keys]

        state_info_varss = [{
            k: tf.placeholder(tf.float32, shape=[None] * (1) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            } for _ in range(self.nactions)]
        print("CATHYWU state info vars", state_info_varss)
        state_info_vars_list = [state_info_vars[k] for state_info_vars
                                in state_info_varss for k in
                                self.policy.state_info_keys]

        dist1 = DiagonalGaussian(1)
        # TODO(cathywu) Can I just create dist1 ad hoc like this?
        lrs = [0 for _ in range(self.nactions)]
        kls = [0 for _ in range(self.nactions)]
        for k in range(self.nactions):
            print("CATHYWU policy", self.policy)
            dist_info_vars = self.policy.dist_info_sym(obs_var,
                                                       state_info_varss[k])
            # TODO(cathywu) split into k 1-by-1 Diagonal covariance matrix
            kls[k] = dist1.kl_sym(old_dist_info_varss[k], dist_info_vars)
            print("CATHYWU lrs", action_vars[k], old_dist_info_varss[k], dist_info_vars)
            lrs[k] = dist1.likelihood_ratio_sym(action_vars[k],
                                              old_dist_info_varss[k],
                                           dist_info_vars, idx=k)

        # FIXME(cathywu) incomplete
        k = 0
        mean_kl = tf.reduce_mean(kls[k])
        # TODO(cathywu) product between adv and ratio of likelihoods
        surr_loss = - tf.reduce_mean(lrs[k] * advantage_vars[k])

        input_list = ext.flatten_list([
                         obs_var,
                         action_vars,
                         advantage_vars,
                     ] + state_info_vars_list + old_dist_info_vars_list)
        print("CATHYWU input_list", input_list)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        # TODO(cathywu) need to compute nactions x advantages
        all_input_values = tuple(ext.flatten_list(ext.extract(
            samples_data,
            "observations", "actions_single", "advantages_single"
        )))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = ext.flatten_list([ext.single(agent_infos[k],
                                                      expand_dims=True)
                                           for k in
                          self.policy.distribution.dist_info_keys])
        # print("CATHYWU state info list", state_info_list)
        # print("CATHYWU dist info list", dist_info_list)
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        logger.log("Computing loss before")
        # print("CATHYWU just before error", all_input_values)
        # print("CATHYWU sizes", [x.shape for x in all_input_values])
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
