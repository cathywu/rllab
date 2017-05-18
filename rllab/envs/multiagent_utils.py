import numpy as np
import scipy


def is_collision(x, eps):
    # https://stackoverflow.com/questions/29608987/
    # pairwise-operations-distance-on-two-lists-in-numpy#29611147
    pairwise_dist = scipy.spatial.distance.cdist(x.T, x.T)
    return np.sum(np.min(pairwise_dist + 1e6 * np.eye(x.shape[1]), axis=1) < eps)


def violates_constraint(pos, upper, lower):
    return np.all((pos > lower) * (pos < upper), axis=0)

