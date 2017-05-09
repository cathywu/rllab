import itertools

import numpy as np
import scipy

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import rllab.misc.logger as logger

NOT_DONE_PENALTY = 1
COLLISION_PENALTY = 10
MAX_RANGE = 10
BOX = 1
LOW = -0.1
HIGH = 0.1

def is_collision(x):
    # https://stackoverflow.com/questions/29608987/
    # pairwise-operations-distance-on-two-lists-in-numpy#29611147
    pairwise_dist = scipy.spatial.distance.cdist(x.T, x.T)
    return np.min(pairwise_dist + 1e6 * np.eye(x.shape[1])) < 0.005


class MultiagentSharedEnv(Env):

    def __init__(self, d=2, k=1, slices=10, horizon=1e6, collisions=False):
        self.d = 2
        self.k = k
        self._slices = slices
        self._horizon = horizon
        self._collisions = collisions

    @property
    def shared_policy(self):
        return True

    @property
    def nagents(self):
        return self.k

    @property
    def per_agent_obsdim(self):
        return int(self.observation_space.flat_dim / self.nagents)

    @property
    def per_agent_actiondim(self):
        return int(self.action_space.flat_dim / self.nagents)

    @property
    def observation_space(self):
        """
        Convention: first dimension indicates per-agent observation space
        :return:
        """
        return Box(low=-np.inf, high=np.inf, shape=(self.nagents, self._slices))

    @property
    def action_space(self):
        """
        Convention: first dimension indicates per-agent observation space
        :return:
        """
        return Box(low=LOW, high=HIGH, shape=(self.nagents, self.d))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._positions = np.random.uniform(-BOX, BOX, size=(self.nagents, self.d))
        self._state = self.get_relative_positions()
        observation = np.copy(self._state)
        # self.plot(agent=0, tag='reset')
        return observation

    def step(self, action):
        action = action.clip(LOW, HIGH)  # Need to manually clip actions
        self._positions = self._positions + action
        self._state = self.get_relative_positions()
        # self.plot(agent=0)

        collisions = np.min(self._state, axis=1) < 0.005 if self._collisions \
            else [False] * self.nagents
        done = False

        local_reward = np.array([- np.sum(
            np.square(self._positions[i, :])) - COLLISION_PENALTY * collisions[
                                     i] for i in range(self.nagents)])
        reward = sum(local_reward)

        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done,
                    local_reward=local_reward, positions=self._positions)

    def get_relative_positions(self):
        """
        Get LIDAR view from each agent
        :return:
        """
        pairs = np.array([x for x in itertools.permutations(self._positions, 2)])
        vecs = pairs[:, 1, :] - pairs[:, 0, :]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0]).reshape([self.nagents, self.nagents - 1])
        a2bin = np.vectorize(lambda x: int((x + np.pi) / (2 * np.pi) * self._slices))
        bins = a2bin(angles)
        dists = np.linalg.norm(vecs, axis=-1).reshape([self.nagents, self.nagents - 1])

        lidar = MAX_RANGE * np.ones(self.observation_space.shape)
        for i in range(self.nagents):
            for j in range(self._slices):
                dvals = dists[i, bins[i, :] == j]
                if len(dvals) > 0:
                    lidar[i, j] = np.min(dvals)
        return lidar

    def render(self):
        print('current state:', self._state)

    def plot(self, agent=0, tag=None):
        import matplotlib.pyplot as plt
        import cmath
        agentx = self._positions[agent, 0]
        agenty = self._positions[agent, 1]

        plt.scatter(self._positions[:, 0], self._positions[:, 1])
        plt.scatter(agentx, agenty, c='red')
        get_angles = np.vectorize(lambda x: -np.pi + 2*np.pi/self._slices*(0.5+x))
        polar = get_angles(range(self._slices))
        to_complex = np.vectorize(lambda x, y: cmath.rect(x, y))
        complex = to_complex(self._state[agent], polar)
        to_rect = np.vectorize(lambda x: (x.real, x.imag))
        x, y = to_rect(complex)
        plt.scatter(x + agentx, y + agenty, marker='+', c='m')
        plt.xlim([-2*BOX, 2*BOX])
        plt.ylim([-2*BOX, 2*BOX])
        if tag is not None:
            fname = 'data/visualization/lidar-%s-%s-agent%s' % (tag, hash(self._positions.tostring()), agent)
        else:
            fname = 'data/visualization/lidar-%s-agent%s' % (hash(self._positions.tostring()), agent)
        plt.savefig(fname)
        plt.clf()
