from rllab.envs.base import Env

class MultiagentEnv(Env):
    def __init__(self, k=1, horizon=1e6, exit_when_done=False, collisions=False,
                 done_epsilon=0.005, done_reward=1, collision_epsilon=0.005,
                 collision_penalty=10, show_actions=True):
        self.k = k
        self._horizon = horizon
        self._collisions = collisions
        self._collision_epsilon = collision_epsilon
        self._collision_penalty = collision_penalty
        # Agents exit when goal is reached
        self._exit_when_done = exit_when_done
        self._done_epsilon = done_epsilon
        self._done_reward = done_reward
        self._show_actions = show_actions

    @property
    def nagents(self):
        return self.k

    @property
    def horizon(self):
        return self._horizon

    @property
    def _exit(self):
        pass
