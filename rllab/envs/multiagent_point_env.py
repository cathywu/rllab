import numpy as np

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import rllab.misc.logger as logger
from rllab.envs import multiagent_utils
from rllab.envs.multiagent_env import MultiagentEnv

NOT_DONE_PENALTY = 1
BOX = 1
LOW = -0.1
HIGH = 0.1


class MultiagentPointEnv(MultiagentEnv):
    def __init__(self, d=2, repeat=1, **kwargs):
        self.d = d
        self._repeat = repeat
        super(MultiagentPointEnv, self).__init__(**kwargs)

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.d, self.nagents))

    @property
    def action_space(self):
        # FIXME(cathywu) reshaping might cause issues here
        # return Box(low=LOW, high=HIGH, shape=(self.d, self.nagents))
        return Box(low=LOW, high=HIGH, shape=(1, self.d * self.nagents))

    def reset(self):
        self._done = np.zeros(self.nagents)
        self._positions = np.random.uniform(0, 2, size=self.observation_space.shape)
        self._state = self._positions  # fully observed
        self._actions = np.zeros(self.action_space.shape).reshape(self.observation_space.shape)
        self._reward = -np.inf  # For plotting only
        self._iter = 0

        self._done_reward = 1  # FIXME(cathywu) remove in cleanup

        observation = np.copy(self._state)
        # self.plot(agent=0, tag='reset')
        return observation

    def step(self, action):
        self._iter += 1
        action_mat = np.reshape(action, self.observation_space.shape)
        self._actions = action_mat  # For plotting only
        total_reward = 0
        for i in range(self._repeat):
            if self._exit_when_done:
                self._positions = self._positions + action_mat * np.tile(
                    (1 - np.isnan(self._done)), [self.d, 1])
            else:
                self._positions = self._positions + action_mat
            self._state = self._positions  # fully observed
            # self.plot(agent=0)

            collision = multiagent_utils.is_collision(self._state, eps=self._collision_epsilon) if self._collisions else False
            # done = collision
            # done = np.all(np.abs(self._state) < 0.02)
            # done = np.all(np.abs(self._state) < 0.01) or collision
            done = False

            # reward = - np.sum(np.square(self._state)) - self._collision_penalty * collision
            # reward = min(np.sum(-np.log(np.abs(self._state))), 100) + 1
            #                 - self._collision_penalty * collision + done * 50
            #          - NOT_DONE_PENALTY
            # reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0))) - \
            #          NOT_DONE_PENALTY
            dist = np.linalg.norm(self._positions, axis=0)
            if self._exit_when_done:
                goal_reward = -dist * (1 - np.isnan(self._done)) + self._done_reward * np.isnan(self._done)
            else:
                goal_reward = -dist
            reward = sum(goal_reward) - self._collision_penalty * collision

            next_observation = np.copy(self._state)
            self._done[dist < self._done_epsilon] = np.nan
            # logger.log('done: {}, collision: {}, reward: {}'.format(done,
            #                                                         collision,
            #                                                         reward))
            total_reward += reward
            self._reward = total_reward  # For plotting only
            # if reward > -3:
            #     self.plot(agent=0)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        self.plot(tag="render")
        # print('current state:', self._state)

    def plot(self, agent=0, tag=None):
        """
        Red: agent
        Blue: all other agents
        Purple crosses: LIDAR "measurements" from agent
        Black: trace of where the agents are coming from
        :param agent:
        :param tag:
        :return:
        """
        import matplotlib.pyplot as plt
        import cmath
        agentx = self._positions[0, agent]
        agenty = self._positions[1, agent]

        plt.scatter(self._positions[0, :], self._positions[1, :])
        done = self._positions[:, np.isnan(self._done)]
        if self._show_actions:
            # Plot trace of where the agents are coming from
            for i in range(self.nagents):
                # actions = np.reshape(self._actions, (self.d, self.nagents))
                dx, dy = self._actions[0, i], self._actions[1, i]
                # NOTE: Action is already applied, so we need to "undo" it
                x, y = self._positions[0, i]-dx, self._positions[1, i]-dy
                plt.arrow(x, y, dx, dy, head_width=0.03, head_length=0.01,
                          fc='k', ec='k')

        # Plot the agents which have completed the task in green
        plt.scatter(done[0, :], done[1, :], c='g')
        # Plot the agent in red
        plt.scatter(agentx, agenty, c='red')

        # Plot the "LIDAR" measurements
        # get_angles = np.vectorize(
        #     lambda x: -np.pi + 2 * np.pi / self._slices * (0.5 + x))
        # polar = get_angles(range(self._slices))
        # to_complex = np.vectorize(lambda x, y: cmath.rect(x, y))
        # complex = to_complex(self._state[agent], polar)
        # to_rect = np.vectorize(lambda x: (x.real, x.imag))
        # x, y = to_rect(complex)
        # plt.scatter(x + agentx, y + agenty, marker='+', c='m')

        # Limit plot size for visual consistency
        plt.xlim([-2*BOX, 2*BOX])
        plt.ylim([-2*BOX, 2*BOX])
        # Plot the overall reward
        plt.text(-2*BOX + 0.1, 2*BOX - 0.3, self._reward, fontsize=12)

        if tag is not None:
            fname = 'data/visualization/lidar-%s-%s-agent%s' % (
                tag, str(self._iter).zfill(5), agent)
        else:
            fname = 'data/visualization/lidar-%s-agent%s' % (str(self._iter).zfill(5), agent)
        plt.savefig(fname)
        plt.clf()

