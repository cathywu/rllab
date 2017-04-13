from rllab.spaces.box import Box as TheanoBox
import tensorflow as tf


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims, flatten=True, size=True):
        if flatten:
            return tf.placeholder(tf.float32, shape=[None] * extra_dims + ([
                self.flat_dim] if size else []), name=name)
            # TODO(cathywu) where is self.flat_dim set?
            # TODO(cathywu) when is flatten uses vs not?
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + (list(
            self.shape) if size else []), name=name)

    @property
    def dtype(self):
        return tf.float32
