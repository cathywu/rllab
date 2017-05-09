import numpy as np
import gym

from rllab.misc.stat_utils import RunningStat
from rllab.envs.base import Env


# class NormalizeObs(gym.Wrapper):
class NormalizeObs(Env):
    def __init__(self, env, center=True, whiten=True, clip=None):
        # super().__init__(env)
        # super().__init__()
        self.center = center
        self.env = env
        self.whiten = whiten
        self.clip = clip
        self._rs = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self.env.horizon

    def reset(self):
        ob = self.env.reset()
        return self._filter(ob)

    def step(self, action):
        next_ob, rew, done, info = self.env.step(action)
        return self._filter(next_ob), rew, done, info

    def _filter(self, x, update=True):
        assert isinstance(x, np.ndarray)
        if self._rs is None:
            self._rs = RunningStat(x.shape)
        if update:
            self._rs.push(x)
        if self.center:
            x = x - self._rs.mean
        if self.whiten:
            x = x / (self._rs.std + 1e-8)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @property
    def nagents(self):
        return self.env.nagents

    @property
    def per_agent_obsdim(self):
        return self.env.per_agent_obsdim

    @property
    def per_agent_actiondim(self):
        return self.env.per_agent_actiondim
