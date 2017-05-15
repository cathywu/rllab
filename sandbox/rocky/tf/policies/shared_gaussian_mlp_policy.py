import numpy as np

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class SharedGaussianMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            std_parametrization='exp'
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        Serializable.quick_init(self, locals())
        self.nagents = env_spec.observation_space.shape[0]
        self.obs_dim = int(env_spec.observation_space.flat_dim / self.nagents)
        self.action_dim = int(env_spec.action_space.flat_dim / self.nagents)
        self.shared_policy = GaussianMLPPolicy(name, env_spec=env_spec,
                                               hidden_sizes=hidden_sizes,
                                               learn_std=learn_std,
                                               init_std=init_std,
                                               adaptive_std=adaptive_std,
                                               std_share_network=std_share_network,
                                               std_hidden_sizes=std_hidden_sizes,
                                               min_std=min_std,
                                               std_hidden_nonlinearity=std_hidden_nonlinearity,
                                               hidden_nonlinearity=hidden_nonlinearity,
                                               output_nonlinearity=output_nonlinearity,
                                               mean_network=mean_network,
                                               std_network=std_network,
                                               std_parametrization=std_parametrization,
                                               obs_dim=self.obs_dim,
                                               action_dim=self.action_dim,
                                               )

    @property
    def vectorized(self):
        return self.shared_policy.vectorized

    def dist_info_sym(self, obs_var, state_info_vars=None, agent=None):
        """
        Notes:
        # N: batch_size
        # obs_var: (N, k * d)
        # obs_var: (N * k, d)
        # shared policy -> (N * k, A)
        # want: (N, k * A)
        # actually outputting: (N * k, A)

        # obs_shape = tf.shape(obs_var)
        # batch_size = obs_shape[0]

        :param obs_var:
        :param state_info_vars:
        :param agent:
        :return:
        """
        agent_obs = tf.reshape(obs_var, [-1, self.obs_dim])

        sym_info = self.shared_policy.dist_info_sym(agent_obs,
                                                 state_info_vars=state_info_vars)
        return sym_info
        # means = tf.reshape(sym_info['mean'], [-1, self.nagents *
        #                                       self.action_dim])
        # log_stds = tf.reshape(sym_info['log_std'], [-1, self.nagents *
        #                                                      self.action_dim])
        # return dict(mean=means, log_std=log_stds)

    @overrides
    def get_action(self, observation, agent=None):
        actions, agent_infos = self.get_actions([observation], agent=agent)
        return actions[0], {k: v[0] for k, v in agent_infos.items()}
        # return self.get_actions([observation], agent=agent)
        # actions, dicts = zip(*[self.shared_policy.get_action(observation[agent, ...])
        #                  for agent in range(self.nagents)])
        # all_dicts = dict(mean=np.array([d['mean'] for d in dicts]),
        #                  log_std=np.array([d['log_std'] for d in dicts]))
        # return np.array(actions), all_dicts

    def get_actions(self, observations, agent=None):
        """
        Input: obs [d, k]
        :param observations:
        :param agent:
        :return:
        """
        actions, dicts = zip(*[self.shared_policy.get_actions(
            [obs[agent, ...] for obs in observations]) for agent in
                               range(self.nagents)])

        # Black magic reshaping. Convention: [N observations, k agents, d actions]
        N = len(observations)
        reshape = lambda x: np.hstack(x).reshape([N, self.nagents, -1])
        actions = reshape(actions)
        all_dicts = dict(mean=reshape([d['mean'] for d in dicts]),
                         log_std=reshape([d['log_std'] for d in dicts]))
        return actions, all_dicts

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        return self.shared_policy.get_reparam_action_sym(obs_var, action_var,
                                                         old_dist_info_vars)

    def log_diagnostics(self, paths):
        self.shared_policy.log_diagnostics(paths)

    @property
    def distribution(self):
        return self.shared_policy._dist

    @property
    def _cached_params(self):
        return self.shared_policy._cached_params

    @property
    def _output_layers(self):
        return self.shared_policy._output_layers

    @property
    def _input_layers(self):
        return self.shared_policy._input_layers

    @property
    def _cached_param_shapes(self):
        return self.shared_policy._cached_param_shapes

    @property
    def _cached_param_dtypes(self):
        return self.shared_policy._cached_param_dtypes

    @property
    def _cached_assign_ops(self):
        return self.shared_policy._cached_assign_ops

    @property
    def _cached_assign_placeholders(self):
        return self.shared_policy._cached_assign_placeholders


