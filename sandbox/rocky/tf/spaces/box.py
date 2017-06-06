from rllab.spaces.box import Box as TheanoBox
import tensorflow as tf


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims, flatten=True, size=None):
        if flatten:
            return tf.placeholder(tf.float32, shape=[None] * extra_dims + ([
                self.flat_dim] if not size else [size]), name=name)
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + (list(
            self.shape) if not size else [size]), name=name)

    @property
    def dtype(self):
        return tf.float32
