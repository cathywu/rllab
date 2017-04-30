from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import numpy as np

NOT_DONE_PENALTY = 1


class MultiactionPointEnv(Env):

    def __init__(self, d=2, k=1, horizon=1e6, collisions=False):
        self.d = d
        self.k = k
        self._horizon = horizon

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.d, 1))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(1, self.d * self.k))

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(self.d, 1))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        action_mat = np.reshape(action, (self.d, self.k))
        overall_action = np.expand_dims(np.sum(action_mat, axis=1), axis=-1)
        self._state = self._state + overall_action

        reward = - np.sum(np.square(self._state)) - NOT_DONE_PENALTY
        # reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0))) - \
        #          NOT_DONE_PENALTY

        done = np.all(np.abs(self._state) < 0.01)
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

