"""
Various routines related to tensorflow

"""

import os
import tensorflow as tf

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

        self.merged_summary = tf.summary.merge_all()
        self.options = tf.RunOptions()
        self.options.output_partition_graphs = True
        self.options.trace_level = tf.RunOptions.SOFTWARE_TRACE
        self.run_metadata = tf.RunMetadata()

    def set_store_frequency(self, freq):
        self.store_frequency = freq

    def set_summary_writer(self, sess):
        self.summary_writer = tf.summary.FileWriter(logdir=self.path, graph=sess.graph)

    def write_summary(self, session, feed_dict, iteration, batch_no):
        # The options flag is needed to obtain profiling information
        summary = session.run(self.merged_summary, feed_dict = feed_dict,
                           options=self.options, run_metadata=self.run_metadata)
        self.summary_writer.add_summary(summary, iteration)
        self.summary_writer.add_run_metadata(self.run_metadata, 'iteration %d batch %d' % (iteration, batch_no))

    def write_weight_histogram(self, weights):
        tf.summary.histogram("weights_in", weights[0])
        for i in range(len(weights) - 1):
            tf.summary.histogram("weights_hidden_%d" % i, weights[i + 1])
        tf.summary.histogram("weights_out", weights[-1])
