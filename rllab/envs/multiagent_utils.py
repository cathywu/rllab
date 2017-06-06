import numpy as np
import scipy


def is_collision(x, eps, mask=None, big_mask=None, lidar=False, local=False,
                 neighbors=None):
    if mask is not None:
        y = x * np.tile(1 - mask, [x.shape[0], 1])
    else:
        y = x

    if lidar:
        nagents = x.shape[0]
        one2n = np.array(range(nagents))
        min_agents = np.argmin(x, axis=1)
        collisions = (x < eps)[np.array(range(min_agents.size)), min_agents]
        nearest_neighbor = neighbors[one2n, min_agents]
        # agents with a lower index are penalized, in the event of a collision
        penalized = (-nagents*2 * (1-collisions) + nearest_neighbor - one2n) > 0
        if local:
            return penalized
        return sum(penalized)
    else:
        # https://stackoverflow.com/questions/29608987/
        # pairwise-operations-distance-on-two-lists-in-numpy#29611147
        pairwise_dist = scipy.spatial.distance.cdist(y.T, y.T)
        # Mask agents marked as nan
        pairwise_dist[np.isnan(pairwise_dist)] = 1e6
        # Exclude diagonal
        pairwise_dist += 1e6 * np.eye(x.shape[1])
        if big_mask is not None:
            pairwise_dist += big_mask * 1e6
        if local:
            return np.min(pairwise_dist, axis=1) < eps
        return np.sum(np.min(pairwise_dist, axis=1) < eps)


def violates_constraint(pos, upper, lower):
    return np.all((pos > lower) * (pos < upper), axis=0)

