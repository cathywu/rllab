from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import numpy as np

NOT_DONE_PENALTY = 1


class NoStateEnv(Env):

    def __init__(self, d=2, k=1, horizon=1e6, collisions=False):
        self.d = d
        self.k = k
        self._horizon = horizon

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(1, 1))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(1, self.d * self.k))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(1, 1))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        action_mat = np.reshape(action, (self.d, self.k))

        reward = - np.sum(np.square(action)) - NOT_DONE_PENALTY
        # reward = - np.sum(np.sqrt(np.sum(np.square(action_mat), axis=0))) - \
        #          NOT_DONE_PENALTY

        done = np.all(np.sum(np.square(action_mat), axis=0) < 0.01)
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

