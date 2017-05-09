

from sandbox.rocky.tf.algos.npo_spatial_discount import NPOSpatialDiscount
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class TRPOSpatialDiscount(NPOSpatialDiscount):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPOSpatialDiscount, self).__init__(optimizer=optimizer, **kwargs)
