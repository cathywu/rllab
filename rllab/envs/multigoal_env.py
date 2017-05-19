import numpy as np
from scipy.linalg import block_diag

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step
import rllab.misc.logger as logger
from rllab.envs import multiagent_utils as ma_utils
from rllab.envs.multiagent_env import MultiagentEnv

NOT_DONE_PENALTY = 1
BOX = 1
LOW = -0.1
HIGH = 0.1


class MultigoalEnv(MultiagentEnv):
    def __init__(self, d=2, ngroups=2, corridor=1, repeat=1,
                 ignore_intra_collisions=False, **kwargs):
        self.d = d
        self._ngroups = ngroups
        self._corridor = corridor
        self._repeat = repeat
        self._ignore_intra_collisions = ignore_intra_collisions
        super(MultigoalEnv, self).__init__(**kwargs)

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.d, self._ngroups * self.nagents))

    @property
    def action_space(self):
        # FIXME(cathywu) reshaping might cause issues here
        # return Box(low=LOW, high=HIGH, shape=(self.d, self.nagents))
        return Box(low=LOW, high=HIGH, shape=(1, self.d * self._ngroups * self.nagents))

    def reset(self):
        self._done = np.zeros(self.nagents * self._ngroups)
        a = self._corridor/2
        rand = np.random.uniform(-a, a, size=self.observation_space.shape)
        self._positions1 = np.tile([0, 1.5], [self.nagents, 1]).T + rand[:, :self.nagents]
        self._positions2 = np.tile([0, -1.5], [self.nagents, 1]).T + rand[:, self.nagents:]
        # self._positions2 = np.tile([1.5, 0], [self.nagents, 1]).T + rand[:, self.nagents:]

        if self._ignore_intra_collisions:
            self._big_mask = block_diag(*[np.ones((self.nagents, self.nagents)) \
                                          for _ in range(self._ngroups)])
        else:
            self._big_mask = None

        self._goal1 = np.tile([0, -1.5], [self.nagents, 1]).T
        self._goal2 = np.tile([0, 1.5], [self.nagents, 1]).T
        # self._goal2 = np.tile([-1.5, 0], [self.nagents, 1]).T
        # Intersection-like constraints
        self._constraints = [
            (np.tile([a, a], [self.nagents, 1]).T,
             np.tile([5, 5], [self.nagents, 1]).T),
            (np.tile([-5, -5], [self.nagents, 1]).T,
             np.tile([-a, -a], [self.nagents, 1]).T),
            (np.tile([a, -5], [self.nagents, 1]).T,
             np.tile([5, -a], [self.nagents, 1]).T),
            (np.tile([-5, a], [self.nagents, 1]).T,
             np.tile([-a, 5], [self.nagents, 1]).T),
        ]
        self._state = np.hstack([self._positions1, self._positions2])  # fully observed
        self._actions = np.zeros(self.action_space.shape).reshape(self.observation_space.shape)
        self._reward = -np.inf  # For plotting only
        self._iter = 0

        observation = np.copy(self._state)
        # self.plot(agent=0, tag='reset')
        return observation

    def step(self, action):
        self._iter += 1
        action_mat = np.reshape(action, self.observation_space.shape)
        self._actions = action_mat  # For plotting only

        total_reward = 0
        for i in range(self._repeat):
            if self._exit_when_done:
                done_mat = np.tile((1 - np.isnan(self._done)), [self.d, 1])
                self._positions1 = self._positions1 + action_mat[:, :self.nagents] * done_mat[:, :self.nagents]
                self._positions2 = self._positions2 + action_mat[:, self.nagents:] * done_mat[:, self.nagents:]
            else:
                self._positions1 = self._positions1 + action_mat[:, :self.nagents]
                self._positions2 = self._positions2 + action_mat[:, self.nagents:]
            self._state = np.hstack([self._positions1, self._positions2])  # fully observed
            # self.plot(agent=0)

            if self._exit_when_done:
                collision = ma_utils.is_collision(self._state,
                                                  eps=self._collision_epsilon,
                                                  mask=self._done,
                                                  big_mask=self._big_mask) \
                    if self._collisions else False
            else:
                collision = ma_utils.is_collision(self._state,
                                                  eps=self._collision_epsilon,
                                                  big_mask=self._big_mask) \
                    if self._collisions else False
            # done = collision
            # done = np.all(np.abs(self._state) < 0.02)
            # done = np.all(np.abs(self._state) < 0.01) or collision
            done = False

            # reward = - np.sum(np.square(self._state)) - self._collision_penalty * collision
            # reward = min(np.sum(-np.log(np.abs(self._state))), 100) + 1
            #                 - self._collision_penalty * collision + done * 50
            #          - NOT_DONE_PENALTY
            # reward = - np.sum(np.sqrt(np.sum(np.square(self._state), axis=0))) - \
            #          NOT_DONE_PENALTY
            # FIXME(cathywu) correct norm wrt goal target
            dist1 = np.linalg.norm(self._positions1 - self._goal1, axis=0)
            dist2 = np.linalg.norm(self._positions2 - self._goal2, axis=0)
            dist = np.concatenate([dist1, dist2])
            if self._exit_when_done:
                goal_reward = -dist * (1 - np.isnan(self._done)) + self._done_reward * np.isnan(self._done)
            else:
                goal_reward = -dist

            # Constraints violation rewards
            violation1 = np.zeros(self.nagents, dtype=np.bool)
            violation2 = np.zeros(self.nagents, dtype=np.bool)
            for constraint in self._constraints:
                lower, upper = constraint
                violation1 = violation1 + ma_utils.violates_constraint(self._positions1, upper, lower)
                violation2 = violation2 + ma_utils.violates_constraint(self._positions2, upper, lower)

            reward = sum(goal_reward) - self._collision_penalty * (collision + sum(violation1) + sum(violation2))

            next_observation = np.copy(self._state)
            self._done[dist < self._done_epsilon] = np.nan
            # logger.log('done: {}, collision: {}, reward: {}'.format(done,
            #                                                         collision,
            #                                                         reward))
            total_reward += reward
            self._reward = total_reward  # For plotting only
            # if reward > -3:
            #     self.plot(agent=0)
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
        import matplotlib.patches as patches
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for constraint in self._constraints:
            lower, upper = constraint
            ax.add_patch(
                patches.Rectangle(lower[:,0],  # (x,y)
                    upper[0, 0] - lower[0, 0],  # width
                    upper[1, 0] - lower[1, 0],  # height
                )
            )
        positions = np.hstack([self._positions1, self._positions2])  # fully observed
        agentx = positions[0, agent]
        agenty = positions[1, agent]

        plt.scatter(positions[0, :self.nagents], positions[1, :self.nagents], color='b')
        plt.scatter(positions[0, self.nagents:], positions[1, self.nagents:], color='m')
        done = positions[:, np.isnan(self._done)]
        if self._show_actions:
            # Plot trace of where the agents are coming from
            for i in range(self.nagents * self._ngroups):
                # actions = np.reshape(self._actions, (self.d, self.nagents))
                dx, dy = self._actions[0, i], self._actions[1, i]
                # NOTE: Action is already applied, so we need to "undo" it
                x, y = positions[0, i]-dx, positions[1, i]-dy
                plt.arrow(x, y, dx, dy, head_width=0.03, head_length=0.01,
                          fc='k', ec='k')

        # Plot the agents which have completed the task in green
        plt.scatter(done[0, :], done[1, :], c='g')
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
                tag, str(self._iter).zfill(5), agent)
        else:
            fname = 'data/visualization/lidar-%s-agent%s' % (str(self._iter).zfill(5), agent)
        plt.savefig(fname)
        plt.clf()
        plt.close()
