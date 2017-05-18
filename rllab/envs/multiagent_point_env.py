import numpy as np
import scipy

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import rllab.misc.logger as logger

NOT_DONE_PENALTY = 1
BOX = 1
LOW = -0.1
HIGH = 0.1

def is_collision(x, eps):
    # https://stackoverflow.com/questions/29608987/
    # pairwise-operations-distance-on-two-lists-in-numpy#29611147
    pairwise_dist = scipy.spatial.distance.cdist(x.T, x.T)
    return np.sum(np.min(pairwise_dist + 1e6 * np.eye(x.shape[1]), axis=1) < eps)


class MultiagentPointEnv(Env):
    def __init__(self, d=2, k=1, horizon=1e6, collisions=False, epsilon=0.005,
                 collision_penalty=10, show_actions=True):
        self.d = d
        self.k = k
        self._horizon = horizon
        self._collisions = collisions
        self._collision_penalty = collision_penalty
        self._epsilon = epsilon
        self._show_actions = show_actions

    @property
    def nagents(self):
        return self.k

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.d, self.nagents))

    @property
    def action_space(self):
        # FIXME(cathywu) reshaping might cause issues here
        # return Box(low=LOW, high=HIGH, shape=(self.d, self.nagents))
        return Box(low=LOW, high=HIGH, shape=(1, self.d * self.nagents))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._state = np.random.uniform(0, 2, size=self.observation_space.shape)
        self._positions = self._state
        self._actions = np.zeros(self.action_space.shape).reshape(self.observation_space.shape)
        self._reward = -np.inf  # For plotting only
        self._iter = 0

        observation = np.copy(self._state)
        # self.plot(agent=0, tag='reset')
        return observation

    def step(self, action):
        self._iter += 1
        # FIXME(cathywu) check reshaping
        action_mat = np.reshape(action, self.observation_space.shape)
        self._actions = action_mat  # For plotting only
        self._state = self._state + action_mat
        self._positions = self._state
        # self.plot(agent=0)

        collision = is_collision(self._state, eps=self._epsilon) if self._collisions else False
        # done = collision
        # done = np.all(np.abs(self._state) < 0.02)
        # done = np.all(np.abs(self._state) < 0.01) or collision
        done = False

        reward = - np.sum(np.square(self._state)) - self._collision_penalty * collision
        # reward = min(np.sum(-np.log(np.abs(self._state))), 100) + 1
        #                 - self._collision_penalty * collision + done * 50
        #          - NOT_DONE_PENALTY
        # reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0))) - \
        #          NOT_DONE_PENALTY
        self._reward = reward  # For plotting only
        # if reward > -3:
        #     self.plot(agent=0)

        next_observation = np.copy(self._state)
        # logger.log('done: {}, collision: {}, reward: {}'.format(done,
        #                                                         collision,
        #                                                         reward))
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
        # done = self._positions[np.isnan(self._done), :]
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
        # plt.scatter(done[:, 0], done[:, 1], c='g')
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
                tag, self._iter, agent)
        else:
            fname = 'data/visualization/lidar-%s-agent%s' % (self._iter, agent)
        plt.savefig(fname)
        plt.clf()

    @property
    def _exit(self):
        pass
