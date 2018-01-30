"""
Permutation Invariant Neural Network Layer
================================================================================

Core functions and classes.
"""

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Conv2D, Lambda
from keras.models import Sequential
import numpy as np
import tensorflow as tf


def permutation_invariant(input_shape, layer_sizes, tuple_dim = 2, reduce_fun = "mean"):
    """
    Implements a permutation invariant layer.

    Args:
    input_shape -- A `tuple` - input shape of one element in a batch.
    layer_sizes -- A `list` of `int`. Sizes of layers in neural network applied to each tuple.
    tuple_dim -- A `int`, size of one tuple.
    reduce_fun -- A `string`, type of function to "average" over all tuples.

    Returns:
    g -- A `Sequential` keras container.
    """
    g = Sequential()
    g.add(Tuples(tuple_dim, input_shape = input_shape)) ## output_shape : input_shape[0] x input_shape[1]**tuple_dim x input_shape[2]*tuple_dim
    prev_num_of_cols = input_shape[2] * tuple_dim
    for layer_size in layer_sizes:
        g.add(Conv2D(filters = layer_size, kernel_size = (1, prev_num_of_cols)))
        prev_num_of_cols = layer_size
    if reduce_fun == "mean":
        lambda_layer = Lambda(lambda x : K.mean(x, axis = 1))
    elif reduce_fun == "max":
        lambda_layer = Lambda(lambda x : K.max(x, axis = 1))
    else:
        raise ValueError("Invalid value for argument `reduce_fun` provided. ")
    g.add(lambda_layer)
    return g

class Tuples(Layer):
    """
    Stack of all possible k-tuple combination of inputs.
    Takes input of shape (batch_size, num_observations, num_features).
    One batch element of input looks like this.

        Input:
        x1
        x2
        ...
        xn

    The features are in the columns. Each row corresponds to one
    observation.

    The output are all possible k-tuples of observations. Each k-tuple is
    represented by one row.

        Output:
        x1 | x1
        x1 | x2
        ...
        x1 | xn
        x2 | x1
        x2 | x2
        ...
        x2 | xn
        ...
        xn | xn
    """

    def __init__(self, tuple_dim = 2, **kwargs):
        self.tuple_dim = tuple_dim
        super(Tuples, self).__init__(**kwargs)

    def create_indices(self, n, k = 2):
        """
        Creates all integer valued coordinate k-tuples in k dimensional hypercube with edge size n.
        for example n = 4, k = 2
        returns [[0, 0], [0, 1], [0, 2], [0, 3],
                 [1, 0], [1, 1], [1, 2], [1, 3],
                 ...
                 [3, 0], [3, 1], [3, 2], [3, 3]]

        Args:
        n -- A `int`, edge size of the hypercube.
        k -- A `int`, dimension of the hypercube.

        Returns:
        indices_n_k -- A `list` of `list` of `int`. Each inner list represents coordinates of one integer point
            in the hypercube.
        """
        if k == 0:
            indices_n_k = [[]]
        else:
            indices_n_k_minus_1 = self.create_indices(n, k-1)
            indices_n_k = [[i] + indices_n_k_minus_1[c] for i in range(n) for c in range(n**(k-1))]

        return indices_n_k


    def build(self, input_shape):
        # Create indexing tuple

        self.gathering_indices = self.create_indices(input_shape[1], self.tuple_dim)
        super(Tuples, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, x):
        stack_of_tuples = K.map_fn(
            fn = lambda z :
                K.concatenate(
                    [K.reshape(
                        K.gather(z, i),
                        shape = (1,-1)
                     )
                     for i in self.gathering_indices
                    ],
                    axis = 0
                ),
            elems = x
        )
        return stack_of_tuples


    def compute_output_shape(self, input_shape):
        output_shape = (
            input_shape[0],
            input_shape[1] ** self.tuple_dim,
            input_shape[2] * self.tuple_dim
        )
        return output_shape


if __name__ == "__main__":
    print("-------------------------------------------------------------------")
    print("Testing Tuples Keras Layer:")
    x = tf.placeholder(shape = (5, 3, 2), dtype = tf.float32) ## 32 experiments in batch, 7 observations in each experiment, 5 feature columns
    tuple_layer = Tuples(tuple_dim = 2, input_shape = (5, 3, 2))
    x_tuppled = tuple_layer(x)
    print(x_tuppled.shape)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    np.random.seed(0)
    feed = {x : np.random.randn(5,3,2)}
    x_eval, x_tuppled_eval = sess.run([x, x_tuppled], feed)
    print("x value is: ", x_eval)
    print("x tuppled value is:", x_tuppled_eval)

    print("-------------------------------------------------------------------")
    print("Testing permutation_invariant function:")
    ##perm_inv_layer = permutation_invariant(input_shape = (5,3,2), layer_sizes = [5,10,5], reduce_fun = "mean")
    x_tuppled
    Conv2D(filters = 10, kernel_size = (1,6))
    Conv2D(filters = 10, kernel_size = (1,6))
