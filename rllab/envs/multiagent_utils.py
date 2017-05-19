import numpy as np
import scipy


def is_collision(x, eps, mask=None, big_mask=None):
    if mask is not None:
        y = x * np.tile(1 - mask, [x.shape[0], 1])
    else:
        y = x
    # https://stackoverflow.com/questions/29608987/
    # pairwise-operations-distance-on-two-lists-in-numpy#29611147
    pairwise_dist = scipy.spatial.distance.cdist(y.T, y.T)
    # Mask agents marked as nan
    pairwise_dist[np.isnan(pairwise_dist)] = 1e6
    # Exclude diagonal
    pairwise_dist += 1e6 * np.eye(x.shape[1])
    if big_mask is not None:
        pairwise_dist += big_mask * 1e6
    return np.sum(np.min(pairwise_dist, axis=1) < eps)


def violates_constraint(pos, upper, lower):
    return np.all((pos > lower) * (pos < upper), axis=0)

