import numpy as np
import scipy

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import rllab.misc.logger as logger

NOT_DONE_PENALTY = 1
COLLISION_PENALTY = 10

def is_collision(x, eps):
    # https://stackoverflow.com/questions/29608987/
    # pairwise-operations-distance-on-two-lists-in-numpy#29611147
    pairwise_dist = scipy.spatial.distance.cdist(x.T, x.T)
    return np.sum(np.min(pairwise_dist + 1e6 * np.eye(x.shape[1]), axis=1) < eps)


class MultiagentPointEnv(Env):

    def __init__(self, d=2, k=1, horizon=1e6, collisions=False, epsilon=0.005):
        self.d = d
        self.k = k
        self._horizon = horizon
        self._collisions = collisions
        self._epsilon = epsilon

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.d, self.k))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(1, self.d * self.k))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(self.d, self.k))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        action_mat = np.reshape(action, (self.d, self.k))
        self._state = self._state + action_mat

        collision = is_collision(self._state, eps=self._epsilon) if self._collisions else False
        # done = collision
        # done = np.all(np.abs(self._state) < 0.02)
        # done = np.all(np.abs(self._state) < 0.01) or collision
        done = False

        reward = - np.sum(np.square(self._state)) - COLLISION_PENALTY * collision
        # reward = min(np.sum(-np.log(np.abs(self._state))), 100) + 1
        #                 - COLLISION_PENALTY * collision + done * 50
        #          - NOT_DONE_PENALTY
        # reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0))) - \
        #          NOT_DONE_PENALTY

        next_observation = np.copy(self._state)
        # logger.log('done: {}, collision: {}, reward: {}'.format(done,
        #                                                         collision,
        #                                                         reward))
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

