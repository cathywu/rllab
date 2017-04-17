


import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.distributions.base import Distribution


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        old_means = old_dist_info["mean"]
        old_log_stds = old_dist_info["log_std"]
        new_means = new_dist_info["mean"]
        new_log_stds = new_dist_info["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = np.square(old_means - new_means) + \
                    np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)
        # more lossy version
        # return TT.sum(
        #     numerator / denominator + TT.log(new_std) - TT.log(old_std ), axis=-1)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, idx=None):
        if idx is not None:
            old_means = tf.expand_dims(old_dist_info_vars["mean"][:, idx],
                                       axis=1)
            old_log_stds = tf.expand_dims(old_dist_info_vars["log_std"][:,
                                          idx], axis=1)
            new_means = tf.expand_dims(new_dist_info_vars["mean"][:, idx],
                                       axis=1)
            new_log_stds = tf.expand_dims(new_dist_info_vars["log_std"][:,
                                          idx], axis=1)
        else:
            old_means = old_dist_info_vars["mean"]
            old_log_stds = old_dist_info_vars["log_std"]
            new_means = new_dist_info_vars["mean"]
            new_log_stds = new_dist_info_vars["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        # TODO(cathywu) Why is there no dimensionality problem here?
        numerator = tf.square(old_means - new_means) + \
                    tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars,
                             new_dist_info_vars, idx=None):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars, idx=idx)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars, idx=idx)

        # TODO(cathywu) remove
        if idx is not None:
            self.new_dist_info_vars[idx] = new_dist_info_vars
            self.old_dist_info_vars[idx] = old_dist_info_vars
            self.logli_new[idx] = logli_new
            self.logli_old[idx] = logli_old

        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars, idx=None):
        if idx is not None:
            x_var = tf.expand_dims(x_var[:, idx], axis=1)
            means = tf.expand_dims(dist_info_vars["mean"][:, idx], axis=1)
            log_stds = tf.expand_dims(dist_info_vars["log_std"][:, idx], axis=1)
            self.means[idx] = means
            self.log_stds[idx] = log_stds
        else:
            # FIXME(cathywu) remove and fix this on the outside
            # x_var = tf.expand_dims(x_var, axis=1)
            means = dist_info_vars["mean"]
            log_stds = dist_info_vars["log_std"]
        zs = (x_var - means) / tf.exp(log_stds)
        # self.zs[idx] = zs
        return - tf.reduce_sum(log_stds, axis=-1) - \
               0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def sample(self, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_info_specs(self):
        return [("mean", (self.dim,)), ("log_std", (self.dim,))]
