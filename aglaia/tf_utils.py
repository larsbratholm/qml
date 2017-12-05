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
