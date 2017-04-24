from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import numpy as np


class MultiagentPointEnv(Env):

    def __init__(self, d=2, k=1, horizon=1e6):
        self.d = d
        self.k = k
        self._horizon = horizon

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.d, self.k))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(self.d, self.k))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(self.d, self.k))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        # TODO(cathywu) slice the actions approrpriately
        reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0)))
        # x, y = self._state
        # reward = - (x ** 2 + y ** 2) ** 0.5
        done = np.all(np.abs(self._state) < 0.01)
        # done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

