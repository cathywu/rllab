import numpy as np
import itertools

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.spaces.discrete import Discrete as TheanoDiscrete
from rllab.spaces.product import Product as TheanoProduct
from rllab.envs.base import Step
import rllab.misc.logger as logger
from rllab.envs import multiagent_utils as ma_utils
from rllab.envs.multiagent_env import MultiagentEnv

NOT_DONE_PENALTY = 1
BOX = 1
LOW = -0.1
HIGH = 0.1


class Sudoku(MultiagentEnv):
    def __init__(self, d=4, **kwargs):
        self.d = d
        super(Sudoku, self).__init__(**kwargs)

    @property
    def observation_space(self):
        # Not used, no state
        return Box(low=-np.inf, high=np.inf, shape=(1, 1))

    @property
    def action_space(self):
        return TheanoProduct(*[TheanoDiscrete(self.d) for _ in range(self.d * self.d)])

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
        # count pairs of violations
        reward = -self.violations(np.reshape(action, [self.d, self.d]))
        self.reward = reward  # For plotting only
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        self.plot(tag="render")
        # print('current state:', self._state)

    def plot(self, agent=0, tag=None):
        pass
