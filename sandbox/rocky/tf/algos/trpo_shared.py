

from sandbox.rocky.tf.algos.npo_shared import NPOShared
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class TRPOShared(NPOShared):
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
        super(TRPOShared, self).__init__(optimizer=optimizer, **kwargs)
