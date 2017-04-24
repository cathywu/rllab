from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.layers_powered import LayersPowered

class MovingTargetRegressor(Serializable):
    """
    Fit the regressor onto a mixture of actual target and current predictions, as a form
    of trust region
    """

    def __init__(self, regressor, mix_fraction):
        Serializable.quick_init(self, locals())
        self.regressor = regressor
        self.mix_fraction = mix_fraction
        self._fitted = False

    def predict(self, x):
        return self.regressor.predict(x)

    def predict_n(self, xs):
        return self.regressor.predict_n(xs)

    def fit(self, xs, ys):
        if self.mix_fraction >= 1.0 or not self._fitted:
            self.regressor.fit(xs, ys)
            self._fitted = True
        else:
            cur_ys = self.predict_n(xs)
            targets = ys * self.mix_fraction + cur_ys * (1 - self.mix_fraction)
            self.regressor.fit(xs, targets)

    def get_param_values(self, **tags):
        return LayersPowered.get_param_values(self.regressor, **tags)

    def set_param_values(self, flattened_params, **tags):
        LayersPowered.set_param_values(self.regressor, flattened_params, **tags)
