from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import numpy as np


class OneStepNoStateEnv(Env):

    def __init__(self, d=2, k=1, horizon=None, collisions=False):
        self.d = d
        self.k = k
        self._horizon = 1

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(1, 1))

    @property
    def action_space(self):
        return Box(low=-10, high=10, shape=(1, self.d * self.k))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._state = np.zeros((1, 1))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        reward = - np.sum(np.square(action))
        done = True  # 1-step env
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

