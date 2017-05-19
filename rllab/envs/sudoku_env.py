import numpy as np
import itertools

from sandbox.rocky.tf.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.envs.base import Step
import rllab.misc.logger as logger
from rllab.envs.multiagent_env import MultiagentEnv


class SudokuEnv(MultiagentEnv):
    def __init__(self, d=4, mask=None, mask_values=None, **kwargs):
        self.d = d
        self._mask = np.array(mask)
        self._mask_values = np.array(mask_values)
        # self._mask = np.array([[0, 1], [1, 3], [2, 0], [3, 2]])
        # self._mask_values = np.array([2, 3, 1, 1])

        super(SudokuEnv, self).__init__(**kwargs)

    @property
    def observation_space(self):
        # Not used, no state
        return Box(low=-np.inf, high=np.inf, shape=(1, 1))

    @property
    def action_space(self):
        return Product(*[Discrete(self.d) for _ in range(self.d * self.d)])

    def violations(self, board):
        """
        Counts the pairs of violations on the (self.d, self.d) board
        :param board:
        :return:
        """
        violations = 0
        for i in range(self.d):
            # column-wise violations
            violations += len(
                [x for x in itertools.combinations(board[:, i], r=2) if
                 x[0] == x[1]])
            # row-wise violations
            violations += len(
                [x for x in itertools.combinations(board[i, :], r=2) if
                 x[0] == x[1]])
        # sub-square violations
        dd = int(np.sqrt(self.d))
        for i in range(dd):
            for j in range(dd):
                violations += len([x for x in itertools.combinations(
                    board[i * dd:i * dd + dd, j * dd:j * dd + dd].flatten(),
                    r=2) if x[0] == x[1]])
        return violations

    def reset(self):
        self._state = np.zeros((1, 1))
        self._actions = self.action_space.sample()
        self._reward = -np.inf  # For plotting only
        self._iter = 0

        observation = np.copy(self._state)
        # self.plot(agent=0, tag='reset')
        return observation

    def step(self, action):
        self._iter += 1

        done = True
        board = np.reshape(action, [self.d, self.d])
        # Use pre-filled values
        board[self._mask[:, 0], self._mask[:, 1]] = self._mask_values
        # count pairs of violations
        reward = -self.violations(board)
        # reward = np.exp(-0.1*self.violations(board))
        self.reward = reward  # For plotting only
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        self.plot(tag="render")
        # print('current state:', self._state)

    def plot(self, agent=0, tag=None):
        pass
