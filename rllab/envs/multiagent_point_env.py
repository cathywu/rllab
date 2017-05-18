from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import numpy as np
import scipy

NOT_DONE_PENALTY = 1

def is_collision(x):
    # https://stackoverflow.com/questions/29608987/
    # pairwise-operations-distance-on-two-lists-in-numpy#29611147
    pairwise_dist = scipy.spatial.distance.cdist(x.T, x.T)
    return np.min(pairwise_dist + 1e6 * np.eye(x.shape[1])) < 0.005


class MultiagentPointEnv(Env):

    def __init__(self, d=2, k=1, horizon=1e6, collisions=False):
        self.d = d
        self.k = k
        self._horizon = horizon
        self._collisions = collisions

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

        reward = - np.sum(np.square(self._state)) - NOT_DONE_PENALTY
        # reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0))) - \
        #          NOT_DONE_PENALTY

        collision = is_collision(self._state) if self._collisions else False
        done = np.all(np.abs(self._state) < 0.01) or collision
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

