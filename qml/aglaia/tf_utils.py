"""
Various routines related to tensorflow

"""

import os
import tensorflow as tf
# Remove packages below when TF 1.10 is released
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops

class TensorBoardLogger(object):
    """
    Helper class for tensorboard functionality

    """

    def __init__(self, path):
        self.path = path
        self.store_frequency = None

    def initialise(self):
        # Create tensorboard directory
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE,
                                     output_partition_graphs=True)
        self.run_metadata = tf.RunMetadata()

    def set_store_frequency(self, freq):
        self.store_frequency = freq

    def set_summary_writer(self, sess):
        self.summary_writer = tf.summary.FileWriter(logdir=self.path, graph=sess.graph)

    def write_summary(self, sess, iteration):

        self.merged_summary = tf.summary.merge_all()
        summary = sess.run(self.merged_summary)
        self.summary_writer.add_summary(summary, iteration)
        self.summary_writer.add_run_metadata(self.run_metadata, 'iteration %d' % (iteration))

    def write_metadata(self, step):
        self.summary_writer.add_run_metadata(self.run_metadata, 'batch %d' % (step))

    def write_weight_histogram(self, weights):
        tf.summary.histogram("weights_in", weights[0])
        for i in range(len(weights) - 1):
            tf.summary.histogram("weights_hidden_%d" % i, weights[i + 1])
        tf.summary.histogram("weights_out", weights[-1])

    def write_cost_summary(self, cost):
        tf.summary.scalar('cost', cost)

def partial_derivatives(y, x):
    """
    Take the partial derivatives of y wrt x. Since the tf.gradients function does a
    sum over the y-dimensions (The output is x.shape), we have to do the derivative
    separately on every single element of y.
    Code modified from https://github.com/tensorflow/tensorflow/issues/675#issuecomment-299729653

    :param y: Rank N Tensor where N >= 1
    :type y: Tensorflow Tensor
    :param x: Rank N Tensor where N >= 0
    :type x: Tensorflow Tensor
    :return: Partial derivatives of shape y.shape + x.shape
    :rtype: Tensorflow Tensor
    """

    y_flat = tf.reshape(y, [-1])
    n = y_flat.shape[0]

    loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=n),
            ]

    _, jacobian = tf.while_loop(
    lambda j, _: j < n,
    lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
    loop_vars)

    # once we have changed how the representation is generated
    return tf.reshape(jacobian.stack(), y.shape.as_list() + x.shape.as_list())

def partial_derivatives_batch(output, inp):
    """
    Take the partial derivatives of output wrt inp. Code is available from
    tensorflow v1.10 and should be replaced at that point.

    :param output: Tensor of shape (n_samples, ...)
    :type output: Tensorflow Tensor
    :param inp: Tensor of shape (n_samples, ...)
    :type inp: Tensorflow Tensor
    :return: Partial derivatives of shape (n_samples,) y.shape[1:] + x.shape[1:]
    :rtype: Tensorflow Tensor
    """

    def loop_fn(i):
        y = array_ops.gather(output, i, axis=1)
        return gradient_ops.gradients(y, inp)[0]

    def for_loop(loop_fn, loop_fn_dtypes, iters):

        flat_loop_fn_dtypes = nest.flatten(loop_fn_dtypes)

        def while_body(i, *ta_list):
            """Body of while loop."""
            fn_output = nest.flatten(loop_fn(i))
            if len(fn_output) != len(flat_loop_fn_dtypes):
                raise ValueError(
                    "Number of expected outputs, %d, does not match the number of "
                    "actual outputs, %d, from loop_fn" % (len(flat_loop_fn_dtypes),
                                                        len(fn_output)))
            outputs = []
            for out, ta in zip(fn_output, ta_list):
                # TODO(agarwal): support returning Operation objects from loop_fn.
                assert isinstance(out, ops.Tensor)
                outputs.append(ta.write(i, array_ops.expand_dims(out, 0)))
            return tuple([i + 1] + outputs)

        ta_list = control_flow_ops.while_loop(
            lambda i, *ta: i < iters, while_body, [0] + [
                tensor_array_ops.TensorArray(dtype, iters)
                for dtype in flat_loop_fn_dtypes
            ])[1:]

        # TODO(rachelim): enable this for sparse tensors
        return nest.pack_sequence_as(loop_fn_dtypes, [ta.concat() for ta in ta_list])

    output_shape = output.shape
    if not output_shape[0].is_compatible_with(inp.shape[0]):
        raise ValueError("Need first dimension of output shape (%s) and inp shape "
                         "(%s) to match." % (output.shape, inp.shape))
    if output_shape.is_fully_defined():
        batch_size = int(output_shape[0])
        output_row_size = output_shape.num_elements() // batch_size
    else:
        output_shape = array_ops.shape(output)
        batch_size = output_shape[0]
        output_row_size = array_ops.size(output) // batch_size
    inp_shape = array_ops.shape(inp)
    # Flatten output to 2-D.
    with ops.control_dependencies(
          [check_ops.assert_equal(batch_size, inp_shape[0])]):
        output = array_ops.reshape(output, [batch_size, output_row_size])


    pfor_output = for_loop(loop_fn, output.dtype,
                                              output_row_size)
    pfor_output = array_ops.reshape(pfor_output,
                                    [output_row_size, batch_size, -1])
    output = array_ops.transpose(pfor_output, [1, 0, 2])
    new_shape = array_ops.concat([output_shape, inp_shape[1:]], axis=0)
    return array_ops.reshape(output, new_shape)
