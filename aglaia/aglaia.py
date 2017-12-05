"""
Main module where all the neural network magic happens
"""

from __future__ import print_function
#from __future__ import absolute_import
import os
import sys
#sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y#, check_array
import tensorflow as tf
#import inverse_dist as inv
#import matplotlib.pyplot as plt
#from sklearn.metrics import r2_score
#from tensorflow.python.framework import ops
#from tensorflow.python.training import saver as saver_lib
#from tensorflow.python.framework import graph_io
#from tensorflow.python.tools import freeze_graph

#TODO relative imports
from utils import is_positive, is_positive_integer, is_positive_integer_or_zero, \
        is_bool, is_string, is_positive_or_zero, InputError
from tf_utils import TensorBoardLogger

class _NN(BaseEstimator, RegressorMixin):

    """
    Parent class for training multi-layered neural networks on molecular or atomic properties via Tensorflow
    """

    def __init__(self, hidden_layer_sizes = [5], l1_reg = 0.0, l2_reg = 0.0001, batch_size = 'auto', learning_rate = 0.001,
                 iterations = 500, tensorboard = False, store_frequency = 200, tf_dtype = tf.float32,
                 activation_function = tf.sigmoid, tensorboard_subdir = os.getcwd() + '/tensorboard', **args):
        """
        :param hidden_layer_sizes: Number of hidden layers. The n'th element represents the number of neurons in the n'th
            hidden layer.
        :type hidden_layer_size: Tuple of integers
        :param l1_reg: L1-regularisation parameter for the neural network weights
        :type l1_reg: float
        :param l2_reg: L2-regularisation parameter for the neural network weights
        :type l2_reg: float
        :param batch_size: Size of minibatches for the ADAM optimizer. If set to 'auto' ``batch_size = min(200,n_samples)``
        :type batch_size: integer
        :param learning_rate: The learning rate in the numerical minimisation.
        :type learning_rate: float
        :param iterations: Total number of iterations that will be carried out during the training process.
        :type iterations: integer
        :param tf_dtype: Accuracy to use for floating point operations in tensorflow. 64 and 'float64' is recognised as tf.float64
            and similar for tf.float32 and tf.float16.
        :type tf_dtype: Tensorflow datatype
        :param activation_function: Activation function to use in the neural network. Currently 'sigmoid', 'tanh', 'elu', 'softplus',
            'softsign', 'relu', 'relu6', 'crelu' and 'relu_x' is supported.
        :type activation_function: Tensorflow datatype
        :param tensorboard: Store summaries to tensorboard or not
        :type tensorboard: boolean
        :param store_frequency: How often to store summaries to tensorboard.
        :type store_frequency: integer
        :param tensorboard_subdir: Directory to store tensorboard data
        :type tensorboard_subdir: string
        """

        # Catch unrecognised passed variables
        if len(args) > 0:
            msg = "Warning: unrecognised input variable(s): "
            msg += ", ".join([str(x for x in args.keys())])
            print(msg)


        # Initialising the parameters
        self._set_hidden_layers_sizes(hidden_layer_sizes)
        self._set_l1_reg(l1_reg)
        self._set_l2_reg(l2_reg)
        self._set_batch_size(batch_size)
        self._set_learning_rate(learning_rate)
        self._set_iterations(iterations)
        self._set_tf_dtype(tf_dtype)
        self._set_tensorboard(tensorboard, store_frequency, tensorboard_subdir)

        if activation_function in ['sigmoid', tf.nn.sigmoid]:
            self.activation_function = tf.nn.sigmoid
        elif activation_function in ['tanh', tf.nn.tanh]:
            self.activation_function = tf.nn.tanh
        elif activation_function in ['elu', tf.nn.elu]:
            self.activation_function = tf.nn.elu
        elif activation_function in ['softplus', tf.nn.softplus]:
            self.activation_function = tf.nn.softplus
        elif activation_function in ['softsign', tf.nn.softsign]:
            self.activation_function = tf.nn.softsign
        elif activation_function in ['relu', tf.nn.relu]:
            self.activation_function = tf.nn.relu
        elif activation_function in ['relu6', tf.nn.relu6]:
            self.activation_function = tf.nn.relu6
        elif activation_function in ['crelu', tf.nn.crelu]:
            self.activation_function = tf.nn.crelu
        elif activation_function in ['relu_x', tf.nn.relu_x]:
            self.activation_function = tf.nn.relu_x
        else:
            raise InputError("Unknown activation function. Got %s" % str(activation_function))


        # Flag to tell if a fit has been made
        self.fit_exist = False

        # Flag to tell if weights have been initialised
        self.initialised = False

        # Placeholder variables
        self.n_features = None
        self.n_samples = None
        self.train_cost = []
        self.test_cost = []
        #self.loaded_model = False
        #self.is_vis_ready = False

    def _set_l1_reg(self, l1_reg):
        if not is_positive_or_zero(l1_reg):
            raise InputError("Expected positive float value for variable 'l1_reg'. Got %s" % str(l1_reg))
        self.l1_reg = l1_reg

    def _set_l2_reg(self, l2_reg):
        if not is_positive_or_zero(l2_reg):
            raise InputError("Expected positive float value for variable 'l2_reg'. Got %s" % str(l2_reg))
        self.l2_reg = l2_reg

    def _set_batch_size(self, batch_size):
        if batch_size != "auto":
            if not is_positive_integer(batch_size):
                raise InputError("Expected 'batch_size' to be a positive integer. Got %s" % str(batch_size))
            elif batch_size == 1:
                raise InputError("batch_size must be larger than 1. Got %s" % str(batch_size))
            self.batch_size = int(batch_size)
        else:
            self.batch_size = batch_size

    def _set_learning_rate(self, learning_rate):
        if not is_positive(learning_rate):
            raise InputError("Expected positive float value for variable learning_rate. Got %s" % str(learning_rate))
        self.learning_rate = float(learning_rate)

    def _set_iterations(self, iterations):
        if not is_positive_integer(iterations):
            raise InputError("Expected positive integer value for variable iterations. Got %s" % str(iterations))
        self.iterations = float(iterations)

    def _set_tf_dtype(self, tf_dtype):
        # 2 == tf.float64 and 1 == tf.float32 for some reason
        # np.float64 recognised as tf.float64 as well
        if tf_dtype in ['64', 64, 'float64', tf.float64]:
            self.tf_dtype = tf.float64
        elif tf_dtype in ['32', 32, 'float32', tf.float32]:
            self.tf_dtype = tf.float32
        elif tf_dtype in ['16', 16, 'float16', tf.float16]:
            self.tf_dtype = tf.float16
        else:
            raise InputError("Unknown tensorflow data type. Got %s" % str(tf_dtype))

    #TODO test
    def _set_hidden_layers_sizes(self, hidden_layer_sizes):
        hidden_layers = []
        try:
            hidden_layer_sizes[0]
        except TypeError:
            raise InputError("'hidden_layer_sizes' must be array-like")
        except IndexError:
            raise InputError("'hidden_layer_sizes' must be non-empty")
        for i,n in enumerate(hidden_layer_sizes):
            if not is_positive_integer_or_zero(n):
                raise InputError("Hidden layer size must be a positive integer. Got %s" % str(n))

            # Ignore layers of size zero
            if int(n) == 0:
                break
            hidden_layers.append(int(n))

        if len(hidden_layers) == 0:
            raise InputError("Hidden layers must be non-zero. Got %s" % str(hidden_layer_sizes))

        self.hidden_layer_sizes = np.asarray(hidden_layers, dtype=int)

    def _set_tensorboard(self, tensorboard, store_frequency, tensorboard_subdir):

        if not is_bool(tensorboard):
            raise InputError("Expected boolean value for variable tensorboard. Got %s" % str(tensorboard))
        self.tensorboard = bool(tensorboard)

        if not self.tensorboard:
            return

        if not is_string(tensorboard_subdir):
            raise InputError('Expected string value for variable tensorboard_subdir. Got %s' % str(self.tensorboard_subdir))

        # TensorBoardLogger will handle all tensorboard related things
        self.tensorboard_logger = TensorBoardLogger(tensorboard_subdir)

        if not is_positive_integer(store_frequency):
            raise InputError("Expected positive integer value for variable store_frequency. Got %s" % str(store_frequency))

        if store_frequency > self.iterations:
            print("Only storing final iteration for tensorboard")
            self.tensorboard_logger.set_store_frequency(self.iterations)
        else:
            self.tensorboard_logger.set_store_frequency(store_frequency)


    # TODO test
    def _init_weight(self, n1, n2, name):
        """
        Generate a tensor of weights of size (n1, n2)

        """

        w = tf.Variable(tf.truncated_normal([n1,n2], stddev = 1.0 / np.sqrt(n2), dtype = self.tf_dtype),
                dtype = self.tf_dtype, name = name)

        return w

    # TODO test
    def _init_bias(self, n, name):
        """
        Generate a tensor of biases of size n.

        """

        b = tf.Variable(tf.zeros([n]), name=name, dtype = self.tf_dtype)

        return b

    #TODO test
    def _generate_weights(self, n_out):
        """
        Generates the weights and the biases, by looking at the size of the hidden layers,
        the number of features in the descriptor and the number of outputs. The weights are initialised from
        a zero centered normal distribution with precision :math:`\\tau = a_{m}`, where :math:`a_{m}` is the number
        of incoming connections to a neuron. Weights larger than two standard deviations from the mean is
        redrawn.

        :param n_out: Number of outputs
        :type n_out: integer
        :return: tuple of weights and biases, each being of length (n_hidden_layers + 1)
        :rtype: tuple
        """

        weights = []
        biases = []

        # Weights from input layer to first hidden layer
        weights.append(self._init_weight(self.hidden_layer_sizes[0], self.n_features, 'weight_in'))
        biases.append(self._init_bias(self.hidden_layer_sizes[0], 'bias_in'))

        # Weights from one hidden layer to the next
        for i in range(1, self.hidden_layer_sizes.size):
            weights.append(self._init_weight(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1], 'weight_hidden_%d' %i))
            biases.append(self._init_bias(self.hidden_layer_sizes[i], 'bias_hidden_%d' % i))

        # Weights from last hidden layer to output layer
        weights.append(self._init_weight(n_out, self.hidden_layer_sizes[-1], 'weight_out'))
        biases.append(self._init_bias(n_out, 'bias_out'))

        return weights, biases

    #TODO test
    def _l2_loss(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l2_loss")

        for i in range(self.hidden_layer_sizes.size):
            reg_term += tf.reduce_sum(tf.square(weights[i]))

        return self.l2_reg * reg_term

    #TODO test
    def _l1_loss(self, weights):
        """
        Creates the expression for L1-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l1_loss")

        for i in range(self.hidden_layer_sizes.size):
            reg_term += tf.reduce_sum(tf.abs(weights[i]))

        return self.l1_reg * reg_term


    def model(self, x, weights, biases):
        """
        Constructs the actual network.

        :param x: Input
        :type x: tf.placeholder of shape (None, n_features)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param biases: Biases used in the network.
        :type biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Output
        :rtype: tf.Variable of size (None, n_targets)
        """

        # Calculate the activation of the first hidden layer
        z = tf.add(tf.matmul(x, tf.transpose(weights[0])), biases[0])
        h = self.activation_function(z)

        # Calculate the activation of the remaining hidden layers
        for i in range(self.hidden_layer_sizes.size-1):
            z = tf.add(tf.matmul(h, tf.transpose(weights[i+1])), biases[i+1])
            h = self.activation_function(z)

        # Calculating the output of the last layer
        z = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1])

        return z

    #TODO test
    def _get_batch_size(self):
        """
        Determines the actual batch size. If set to auto, the batch size will be set to 200.
        If the batch size is larger than the number of samples, it is truncated and a warning
        is printed.

        :return: Batch size
        :rtype: integer
        """

        if self.batch_size == 'auto':
            return min(100, self.n_samples)
        else:
            if self.batch_size > self.n_samples:
                print("Warning: batch_size larger than sample size. It is going to be clipped")
                return min(self.nsamples, self.batch_size)
            else:
                return self.batch_size

    def plot_cost(self, test = True, filename = ''):
        """
        Plots the value of the cost function as a function of the iterations.

        :param test: Whether to plot the accuracy on a test set as well if it was given in the fit
        :type test: boolean
        :param filename: File to save the plot to. If '' the plot is shown instead of saved.
        :type filename: string
        """

        try:
            import pandas as pd
            import seaborn as sns
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'pandas'")

        sns.set()
        df = pd.DataFrame()
        df["Iterations"] = range(len(self.train_cost))
        df["Training cost"] = self.train_cost
        sns.lmplot('Iterations', 'Training cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)
        if test:
            df["Testing cost"] = self.test_cost
            sns.lmplot('Iterations', 'Testing cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)

        if filename == '':
            plt.show()
        elif is_string(filename):
            plt.save(filename)
        else:
            raise InputError("Wrong data type of variable 'filename'. Expected string")

    def correlation_plot(self, y_nn, y_true, filename = ''):
        """
        Creates a correlation plot between predictions and true values.

        :param y_predicted: Values predicted by the neural net
        :type y_predicted: list
        :param y_true: True values
        :type y_true: list
        :param filename: File to save the plot to. If '' the plot is shown instead of saved.
                         If the dimensionality of y is higher than 1, the filename will be prefixed
                         by the dimension.
        :type filename: string
        """

        try:
            import pandas as pd
            import seaborn as sns
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'pandas'")

        if y_nn.shape != y_true.shape:
            raise InputError("Shape mismatch between predicted and true values. %s and %s" % (str(y_nn.shape), str(y_true.shape)))

        if y_nn.ndim == 1:
            df = pd.DataFrame()
            df["Predictions"] = y_nn
            df["True"] = y_true
            sns.set()
            lm = sns.lmplot('True', 'Predictions', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
            if filename == '':
                plt.show()
            elif is_string(filename):
                plt.save(filename)
            else:
                raise InputError("Wrong data type of variable 'filename'. Expected string")
        else:
            for i in range(y_nn.ndim):
                df = pd.DataFrame()
                df["Predictions"] = y_nn[:,i]
                df["True"] = y_true[:,i]
                sns.set()
                lm = sns.lmplot('True', 'Predictions', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
                if filename == '':
                    plt.show()
                elif is_string(filename):
                    tokens = filename.split("/")
                    file_ = str(i) + "_" + tokens[-1]
                    if len(tokens) > 1:
                        file_ = "/".join(tokens[:-1]) + "/" + file_
                    plt.save(file_)
                else:
                    raise InputError("Wrong data type of variable 'filename'. Expected string")

    #TODO test
    def _get_batch_size(self):
        """
        This function is called at fit time to automatically get the batch size.
        If it is a user set value, it checks whether it is a reasonable value.

        :return: int

        """
        if self.batch_size == 'auto':
            batch_size = min(100, self.n_samples)
        elif self.batch_size > self.n_samples:
            print("Warning: Got 'batch_size' larger than sample size. It is going to be clipped")
            batch_size = max(self.batch_size, self.n_samples)

        return batch_size

# Molecular Representation Single Property
class MRMP(_NN):
    """
    Neural network for predicting single properties, such as energies, using molecular representations.
    """

    def __init__(self, **args):
        """
        Molecular descriptors is used as input to a single or multi layered feed-forward neural network with a single output.
        This class inherits from the _NN class and all inputs not unique to the MRMP class is passed to the _NN
        parent.

        """

        super(MRMP,self).__init__(**args)

    #TODO test
    def fit(self, x, y):
        """
        Fit the neural network to molecular representations x and target y.

        :param x: Input data with samples in the rows and features in the columns.
        :type x: array
        :param y: Target values for each sample.
        :type y: array

        """

        return self._fit(x, y)

    def _fit(self, x, y):

        # Check that X and y have correct shape
        x, y = check_X_y(x, y, multi_output = False, y_numeric = True, warn_on_dtype = True)

        # reshape to tensorflow friendly shape
        y = np.atleast_2d(y).T

        # Flag that a model has been trained
        self.fit_exist = True

        # Useful quantities
        self.n_features = x.shape[1]
        self.n_samples = x.shape[0]
        #self.n_atoms = int(X.shape[1]/3)

        # Set the batch size
        batch_size = self._get_batch_size()

        # Initial set up of the NN
        with tf.name_scope("Data"):
            tf_x = tf.placeholder(self.tf_dtype, [None, self.n_features], name="Descriptors")
            tf_y = tf.placeholder(self.tf_dtype, [None, 1], name="Properties")

        # Either initialise the weights and biases or restart training from wherever it was stopped
        with tf.name_scope("Weights"):
            weights, biases = self._generate_weights(n_out = 1)

            # Log weights for tensorboard
            if self.tensorboard:
                tf.summary.histogram("weights_in", weights[0])
                for i in range(self.hidden_layer_sizes.size - 1):
                    tf.summary.histogram("weights_hidden_%d" % i, weights[i + 1])
                tf.summary.histogram("weights_out", weights[-1])

        with tf.name_scope("Model"):
            y_pred = self.model(tf_x, weights, biases)


        with tf.name_scope("Cost_func"):
            cost = self.cost(y_pred, tf_y, weights)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        #self.initialised = True

        if self.tensorboard:
            self.tensorboard_logger.initialise()

        # This is the total number of batches in which the training set is divided
        n_batches = self.n_samples // self.batch_size

        # Running the graph
        with tf.Session() as sess:
            if self.tensorboard:
                self.tensorboard_logger.set_summary_writer(sess)

            sess.run(init)

            for i in range(self.max_iter):
                # This will be used to calculate the average cost per iteration
                avg_cost = 0
                # Learning over the batches of data
                for i in range(n_batches):
                    batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
                    batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
                    opt, c = sess.run([optimizer, cost], feed_dict={X_train: batch_x, Y_train: batch_y})
                    avg_cost += c / n_batches

                    if self.tensorboard:
                        if iter % self.print_step == 0:
                            # The options flag is needed to obtain profiling information
                            summary = sess.run(merged_summary, feed_dict={X_train: batch_x, Y_train: batch_y},
                                               options=options, run_metadata=run_metadata)
                            summary_writer.add_summary(summary, iter)
                            summary_writer.add_run_metadata(run_metadata, 'iteration %d batch %d' % (iter, i))

                self.trainCost.append(avg_cost)
#
#            # Saving the weights for later re-use
#            self.all_weights = []
#            self.all_biases = []
#            for ii in range(len(weights)):
#                self.all_weights.append(sess.run(weights[ii]))
#                self.all_biases.append(sess.run(biases[ii]))


    def cost(self, y_pred, y, weights):
        """
        Constructs the cost function

        :param y_pred: Predicted output
        :type y_pred: tf.Variable of size (None, 1)
        :param y: True output
        :type y: tf.placeholder of shape (None, 1)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Cost
        :rtype: tf.Variable of size (1,)
        """

        err = tf.square(tf.subtract(y,y_pred))
        loss = tf.reduce_mean(err, name="loss")
        cost = loss
        if self.l2_reg > 0:
            l2_loss = self._l2_loss(weights)
            cost = cost + l2_loss
        if self.l1_reg > 0:
            l1_loss = self._l1_loss(weights)
            cost = cost + l1_loss

        return cost

if __name__ == "__main__":
    lol = MRMP(tensorboard = True)
    x = np.random.random((1000,2))
    y = np.random.random((1000,))
    lol.fit(x,y)
