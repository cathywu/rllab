import numpy as np
from gym import spaces
from sandbox.rocky.tf.spaces.box import Box


def space_to_dist_dim(space):
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Tuple):
        return sum(map(space_to_dist_dim, space.spaces))
    else:
        raise NotImplementedError("Unsupported space type: {}".format(space.__class__))


def space_to_flat_dim(space):
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, Box):
        # TODO(cathywu) backport the rest of these utils to non-gym spaces?
        return space.flat_dim
    elif isinstance(space, spaces.Tuple):
        return sum(map(space_to_flat_dim, space.spaces))
    else:
        raise NotImplementedError("Unsupported space type: {}".format(space.__class__))


def space_to_free_dim(space):
    if isinstance(space, spaces.Discrete):
        return 0
    elif isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Tuple):
        return sum(map(space_to_free_dim, space.spaces))
    else:
        raise NotImplementedError("Unsupported space type: {}".format(space.__class__))


def flatten_n(space, xs):
    if isinstance(space, spaces.Discrete):
        ret = np.zeros((len(xs), space.n), dtype=np.int)
        ret[np.arange(len(xs)), np.asarray(xs)] = 1
        return ret
    elif isinstance(space, spaces.Box):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))
    elif isinstance(space, spaces.Tuple):
        xs_regrouped = [[x[i] for x in xs] for i in range(len(xs[0]))]
        flat_regrouped = [flatten_n(c, xi) for c, xi in zip(space.spaces, xs_regrouped)]
        return np.concatenate(flat_regrouped, axis=-1)
    else:
        raise NotImplementedError("Unsupported space type: {}".format(space.__class__))
