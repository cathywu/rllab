import itertools

import numpy as np

from rllab.envs.base import Env
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step

NOT_DONE_PENALTY = 1
COLLISION_PENALTY = 10
MAX_RANGE = 10
BOX = 1
LOW = -0.1
HIGH = 0.1


class MultiagentSharedEnv(Env):
    def __init__(self, d=2, k=1, slices=10, exit_when_done=False, horizon=1e6,
                 done_epsilon=0.005, collisions=False, collision_epsilon=0.005,
                 show_actions=True):
        self.d = 2
        self.k = k
        self._slices = slices
        self._horizon = horizon
        self._collisions = collisions
        self._collision_epsilon = collision_epsilon
        self._done_epsilon = done_epsilon
        # Agents exit when goal is reached
        self._exit_when_done = exit_when_done
        self._show_actions = show_actions

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
        return Box(low=-np.inf, high=np.inf,
                   shape=(self.nagents, self.d + self._slices))

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
        self._done = np.zeros(self.nagents)
        self._positions = np.random.uniform(-BOX, BOX,
                                            size=(self.nagents, self.d))
        self._state = self.get_relative_positions()
        # For plotting only
        self._reward = -np.inf
        self._actions = np.zeros(self.action_space.shape)
        self._show_actions = True  # FIXME(cathywu) Remove in cleanup
        self._iter = 0

        observation = np.copy(np.hstack((self._positions, self._state)))
        # self.plot(agent=0, tag='reset')
        return observation

    def step(self, action):
        self._iter += 1
        action = action.clip(LOW, HIGH)  # Need to manually clip actions
        self._actions = action  # For plotting only
        if self._exit_when_done:
            self._positions = self._positions + action * np.tile(
                (1 - np.isnan(self._done)), [self.d, 1]).T
        else:
            self._positions = self._positions + action
        self._state = self.get_relative_positions()
        # self.plot(agent=0)

        collisions = np.min(self._state, axis=1) < self._collision_epsilon if self._collisions \
            else np.array([False] * self.nagents)
        done = False

        if self._exit_when_done:
            dist = np.linalg.norm(self._positions, axis=1) * (
                1 - np.isnan(self._done))
            local_reward = -dist - COLLISION_PENALTY * collisions + 1 * np.isnan(
                self._done)
        else:
            dist = np.linalg.norm(self._positions, axis=1)
            local_reward = -dist - COLLISION_PENALTY * collisions
        reward = sum(local_reward)
        self._reward = reward  # For plotting only
        # if np.sum(dist < self._done_epsilon) >= 4:
        #     self.plot(agent=0)
        #     import ipdb
        #     ipdb.set_trace()

        next_observation = np.hstack([self._positions, self._state])
        self._done[dist < self._done_epsilon] = np.nan
        return Step(observation=next_observation, reward=reward, done=done,
                    local_reward=local_reward, positions=self._positions)

    def get_relative_positions(self):
        """
        Get LIDAR view from each agent
        :return:
        """
        done_perm = np.array([x for x in itertools.permutations(self._done, 2)])
        done_mask = np.isnan((done_perm[:, 1] - done_perm[:, 0]).reshape(
            [self.nagents, self.nagents - 1]))
        if len(self._positions.shape) == 3:
            if self._positions.shape[0] == 1:
                self._positions = self._positions[0, ...]
                self._actions = self._actions[0, ...]
            else:
                return NotImplementedError
        pairs = np.array(
            [x for x in itertools.permutations(self._positions, 2)])
        # Pairwise vectors between agents
        vecs = pairs[:, 1, :] - pairs[:, 0, :]
        # Corresponding angles between agents
        angles = np.arctan2(vecs[:, 1], vecs[:, 0]).reshape(
            [self.nagents, self.nagents - 1])
        # Bin the angles into discretized slices of the view
        a2bin = np.vectorize(
            lambda x: int((x + np.pi) / (2 * np.pi) * self._slices))
        bins = a2bin(angles)
        # Corresponding distances between agents
        dists = np.linalg.norm(vecs, axis=-1).reshape(
            [self.nagents, self.nagents - 1])
        if self._exit_when_done:
            dists += MAX_RANGE * done_mask

        lidar = MAX_RANGE * np.ones((self.nagents, self._slices))
        for i in range(self.nagents):
            for j in range(self._slices):
                dvals = dists[i, bins[i, :] == j]
                if len(dvals) > 0:
                    lidar[i, j] = min(MAX_RANGE, np.min(dvals))
        return lidar

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
        import cmath
        agentx = self._positions[agent, 0]
        agenty = self._positions[agent, 1]

        plt.scatter(self._positions[:, 0], self._positions[:, 1])
        done = self._positions[np.isnan(self._done), :]
        if self._show_actions:
            # Plot trace of where the agents are coming from
            for i in range(self.nagents):
                dx, dy = self._actions[i, 0], self._actions[i, 1]
                # NOTE: Action is already applied, so we need to "undo" it
                x, y = self._positions[i, 0]-dx, self._positions[i, 1]-dy
                plt.arrow(x, y, dx, dy, head_width=0.03, head_length=0.01,
                          fc='k', ec='k')

        # Plot the agents which have completed the task in green
        plt.scatter(done[:, 0], done[:, 1], c='g')
        # Plot the agent in red
        plt.scatter(agentx, agenty, c='red')

        # Plot the "LIDAR" measurements
        get_angles = np.vectorize(
            lambda x: -np.pi + 2 * np.pi / self._slices * (0.5 + x))
        polar = get_angles(range(self._slices))
        to_complex = np.vectorize(lambda x, y: cmath.rect(x, y))
        complex = to_complex(self._state[agent], polar)
        to_rect = np.vectorize(lambda x: (x.real, x.imag))
        x, y = to_rect(complex)
        plt.scatter(x + agentx, y + agenty, marker='+', c='m')

        # Limit plot size for visual consistency
        plt.xlim([-2*BOX, 2*BOX])
        plt.ylim([-2*BOX, 2*BOX])
        # Plot the overall reward
        plt.text(-2*BOX + 0.1, 2*BOX - 0.3, self._reward, fontsize=12)

        if tag is not None:
            fname = 'data/visualization/lidar-%s-%s-agent%s' % (
                tag, self._iter, agent)
        else:
            fname = 'data/visualization/lidar-%s-agent%s' % (self._iter, agent)
        plt.savefig(fname)
        plt.clf()

    @property
    def _exit(self):
        pass
