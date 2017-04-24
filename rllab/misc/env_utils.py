import gym

from rllab.envs.env_spec import EnvSpec


def get_spec(env):
    if isinstance(env, gym.Env):
        return EnvSpec(
            observation_space=env.observation_space,
            action_space=env.action_space,
            horizon=env.spec.timestep_limit
        )
    else:
        import ipdb; ipdb.set_trace()
        raise NotImplementedError

