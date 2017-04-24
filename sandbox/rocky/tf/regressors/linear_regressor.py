import numpy as np


class LinearRegressor(object):
    def __init__(
            self,
            input_size,
            output_size,
            reg_coeff=1e-5
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.reg_coeff = reg_coeff
        self.coeffs = np.zeros((input_size + 1, output_size))

    def fit(self, xs, ys):
        N = len(xs)
        # add bias
        xs = np.concatenate([xs, np.ones((N, 1))], axis=-1)
        reg_coeff = self.reg_coeff
        xTx = xs.T.dot(xs)
        eye = np.identity(xs.shape[1])
        xTy = xs.T.dot(ys)
        for _ in range(5):
            self.coeffs = np.linalg.lstsq(
                xTx + reg_coeff * eye,
                xTy
            )[0]
            if not np.any(np.isnan(self.coeffs)):
                break
            reg_coeff *= 10

    def predict_n(self, xs):
        N = len(xs)
        # add bias
        xs = np.concatenate([xs, np.ones((N, 1))], axis=-1)
        return xs.dot(self.coeffs)
