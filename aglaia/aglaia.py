"""
Main module where all the neural network magic happens
"""

from __future__ import print_function
#from __future__ import absolute_import
import os
import sys
#sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import numpy as np
import tensorflow as tf
#from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
#import inverse_dist as inv
#from tensorflow.python.framework import ops
#from tensorflow.python.training import saver as saver_lib
#from tensorflow.python.framework import graph_io
#from tensorflow.python.tools import freeze_graph

#from .utils import is_positive, is_positive_integer, is_positive_integer_or_zero, \
#       is_bool, is_string, is_positive_or_zero, InputError, ceil
from .utils import InputError, ceil, is_positive_or_zero, is_positive_integer, is_positive, \
        is_bool, is_positive_integer_or_zero, is_string, is_positive_integer_array, is_array_like
from .tf_utils import TensorBoardLogger

class _NN(object):

    """
    Parent class for training multi-layered neural networks on molecular or atomic properties via Tensorflow
    """

    def __init__(self, hidden_layer_sizes = (5,), l1_reg = 0.0, l2_reg = 0.0001, batch_size = 'auto', learning_rate = 0.001,
                 iterations = 500, tensorboard = False, store_frequency = 200, tf_dtype = tf.float32, scoring_function = 'mae',
                 activation_function = tf.sigmoid, optimiser=tf.train.AdamOptimizer, beta1=0.9, beta2=0.999, epsilon=1e-08,
                 rho=0.95, initial_accumulator_value=0.1, initial_gradient_squared_accumulator_value=0.1,
                 l1_regularization_strength=0.0,l2_regularization_strength=0.0,
                 tensorboard_subdir = os.getcwd() + '/tensorboard', **kwargs):
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
        :param scoring_function: Scoring function to use. Available choices are 'mae', 'rmse', 'r2'.
        :type scoring_function: string
        :param activation_function: Activation function to use in the neural network. Currently 'sigmoid', 'tanh', 'elu', 'softplus',
            'softsign', 'relu', 'relu6', 'crelu' and 'relu_x' is supported.
        :type activation_function: Tensorflow datatype
        :param beta1: parameter for AdamOptimizer
        :type beta1: float
        :param beta2: parameter for AdamOptimizer
        :type beta2: float
        :param epsilon: parameter for AdadeltaOptimizer
        :type epsilon: float
        :param rho: parameter for AdadeltaOptimizer
        :type rho: float
        :param initial_accumulator_value: parameter for AdagradOptimizer
        :type initial_accumulator_value: float
        :param initial_gradient_squared_accumulator_value: parameter for AdagradDAOptimizer
        :type initial_gradient_squared_accumulator_value: float
        :param l1_regularization_strength: parameter for AdagradDAOptimizer
        :type l1_regularization_strength: float
        :param l2_regularization_strength: parameter for AdagradDAOptimizer
        :type l2_regularization_strength: float
        :param tensorboard: Store summaries to tensorboard or not
        :type tensorboard: boolean
        :param store_frequency: How often to store summaries to tensorboard.
        :type store_frequency: integer
        :param tensorboard_subdir: Directory to store tensorboard data
        :type tensorboard_subdir: string
        """

        super(_NN,self).__init__()

        # Catch unrecognised passed variables
        if len(kwargs) > 0:
            msg = "Warning: unrecognised input variable(s): "
            msg += ", ".join([str(x for x in kwargs.keys())])
            print(msg)


        # Initialising the parameters
        self._set_hidden_layers_sizes(hidden_layer_sizes)
        self._set_l1_reg(l1_reg)
        self._set_l2_reg(l2_reg)
        self._set_batch_size(batch_size)
        self._set_learning_rate(learning_rate)
        self._set_iterations(iterations)
        self._set_tf_dtype(tf_dtype)
        self._set_scoring_function(scoring_function)
        self._set_tensorboard(tensorboard, store_frequency, tensorboard_subdir)
        self._set_activation_function(activation_function)

        # Placeholder variables
        self.n_features = None
        self.n_samples = None
        self.training_cost = []
        self.session = None

        # Setting the optimiser
        self._set_optimiser_param(beta1, beta2, epsilon, rho, initial_accumulator_value,
                                  initial_gradient_squared_accumulator_value, l1_regularization_strength,
                                  l2_regularization_strength)

        self.optimiser = self._set_optimiser_type(optimiser)

    def _set_activation_function(self, activation_function):
        """
        This function sets which activation function will be used in the model.

        :param activation_function: name of the activation function to use
        :type activation_function: string or tf class
        :return: None
        """
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

    def _set_l1_reg(self, l1_reg):
        """
        This function sets the value of the l1 regularisation that will be used on the weights in the model.

        :param l1_reg: l1 regularisation on the weights
        :type l1_reg: float
        :return: None
        """
        if not is_positive_or_zero(l1_reg):
            raise InputError("Expected positive float value for variable 'l1_reg'. Got %s" % str(l1_reg))
        self.l1_reg = l1_reg

    def _set_l2_reg(self, l2_reg):
        """
         This function sets the value of the l2 regularisation that will be used on the weights in the model.

         :param l2_reg: l2 regularisation on the weights
         :type l2_reg: float
         :return: None
         """
        if not is_positive_or_zero(l2_reg):
            raise InputError("Expected positive float value for variable 'l2_reg'. Got %s" % str(l2_reg))
        self.l2_reg = l2_reg

    def _set_batch_size(self, batch_size):
        """
        This function sets the value of the batch size. The value of the batch size will be checked again once the data
        set is available, to make sure that it is sensible. This will be done by the function _get_batch_size.

        :param batch_size: size of the batch size.
        :type batch_size: float
        :return: None
        """
        if batch_size != "auto":
            if not is_positive_integer(batch_size):
                raise InputError("Expected 'batch_size' to be a positive integer. Got %s" % str(batch_size))
            elif batch_size == 1:
                raise InputError("batch_size must be larger than 1. Got %s" % str(batch_size))
            self.batch_size = int(batch_size)
        else:
            self.batch_size = batch_size

    def _set_learning_rate(self, learning_rate):
        """
        This function sets the value of the learning that will be used by the optimiser.

        :param learning_rate: step size in the optimisation algorithms
        :type l1_reg: float
        :return: None
        """
        if not is_positive(learning_rate):
            raise InputError("Expected positive float value for variable learning_rate. Got %s" % str(learning_rate))
        self.learning_rate = float(learning_rate)

    def _set_iterations(self, iterations):
        """
        This function sets the number of iterations that will be carried out by the optimiser.

        :param iterations: number of iterations
        :type l1_reg: int
        :return: None
        """
        if not is_positive_integer(iterations):
            raise InputError("Expected positive integer value for variable iterations. Got %s" % str(iterations))
        self.iterations = int(iterations)

    # TODO check that the estimators actually use this
    def _set_tf_dtype(self, tf_dtype):
        """
        This sets what data type will be used in the model.

        :param tf_dtype: data type
        :type tf_dtype: string or tensorflow class or int
        :return: None
        """
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

    def _set_optimiser_param(self, beta1, beta2, epsilon, rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                             l1_regularization_strength, l2_regularization_strength):
        """
        This function sets all the parameters that are required by all the optimiser functions. In the end, only the parameters
        for the optimiser chosen will be used.

        :param beta1: parameter for AdamOptimizer
        :type beta1: float
        :param beta2: parameter for AdamOptimizer
        :type beta2: float
        :param epsilon: parameter for AdadeltaOptimizer
        :type epsilon: float
        :param rho: parameter for AdadeltaOptimizer
        :type rho: float
        :param initial_accumulator_value: parameter for AdagradOptimizer
        :type initial_accumulator_value: float
        :param initial_gradient_squared_accumulator_value: parameter for AdagradDAOptimizer
        :type initial_gradient_squared_accumulator_value: float
        :param l1_regularization_strength: parameter for AdagradDAOptimizer
        :type l1_regularization_strength: float
        :param l2_regularization_strength: parameter for AdagradDAOptimizer
        :type l2_regularization_strength: float
        :return: None
        """
        if not is_positive(beta1) and not is_positive(beta2):
            raise InputError("Expected positive float values for variable beta1 and beta2. Got %s and %s." % (str(beta1),str(beta2)))
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

        if not is_positive(epsilon):
            raise InputError("Expected positive float value for variable epsilon. Got %s" % str(epsilon))
        self.epsilon = float(epsilon)

        if not is_positive(rho):
            raise InputError("Expected positive float value for variable rho. Got %s" % str(rho))
        self.rho = float(rho)

        if not is_positive(initial_accumulator_value) and not is_positive(initial_gradient_squared_accumulator_value):
            raise InputError("Expected positive float value for accumulator values. Got %s and %s" %
                             (str(initial_accumulator_value), str(initial_gradient_squared_accumulator_value)))
        self.initial_accumulator_value = float(initial_accumulator_value)
        self.initial_gradient_squared_accumulator_value = float(initial_gradient_squared_accumulator_value)

        if not is_positive_or_zero(l1_regularization_strength) and not is_positive_or_zero(l2_regularization_strength):
            raise InputError("Expected positive or zero float value for regularisation variables. Got %s and %s" %
                             (str(l1_regularization_strength), str(l2_regularization_strength)))
        self.l1_regularization_strength = float(l1_regularization_strength)
        self.l2_regularization_strength = float(l2_regularization_strength)

    def _set_optimiser_type(self, optimiser):
        """
        This function sets which numerical optimisation algorithm will be used for training.

        :param optimiser: Optimiser
        :type optimiser: string or tf class
        :return: tf optimiser to use
        :rtype: tf class
        """
        self.AdagradDA = False
        if optimiser in ['AdamOptimizer', tf.train.AdamOptimizer]:
            optimiser_type = tf.train.AdamOptimizer
        elif optimiser in ['AdadeltaOptimizer', tf.train.AdadeltaOptimizer]:
            optimiser_type = tf.train.AdadeltaOptimizer
        elif optimiser in ['AdagradOptimizer', tf.train.AdagradOptimizer]:
            optimiser_type = tf.train.AdagradOptimizer
        elif optimiser in ['AdagradDAOptimizer', tf.train.AdagradDAOptimizer]:
            optimiser_type = tf.train.AdagradDAOptimizer
            self.AdagradDA = True
        elif optimiser in ['GradientDescentOptimizer', tf.train.GradientDescentOptimizer]:
            optimiser_type = tf.train.GradientDescentOptimizer
        else:
            raise InputError("Unknown optimiser. Got %s" % str(optimiser))

        return optimiser_type

    def _set_optimiser(self):
        """
        This function instantiates an object from the optimiser class that has been selected by the user. It also sets
        the parameters for the optimiser.

        :return: Optimiser with set parameters
        :rtype: object of tf optimiser class
        """
        self.AdagradDA = False
        if self.optimiser in ['AdamOptimizer', tf.train.AdamOptimizer]:
            optimiser_obj = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                                                    epsilon=self.epsilon)
        elif self.optimiser in ['AdadeltaOptimizer', tf.train.AdadeltaOptimizer]:
             optimiser_obj = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon)
        elif self.optimiser in ['AdagradOptimizer', tf.train.AdagradOptimizer]:
             optimiser_obj = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=self.initial_accumulator_value)
        elif self.optimiser in ['AdagradDAOptimizer', tf.train.AdagradDAOptimizer]:
            self.global_step = tf.placeholder(dtype=tf.int64)
            optimiser_obj = tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate, global_step=self.global_step,
                                                         initial_gradient_squared_accumulator_value=self.initial_gradient_squared_accumulator_value,
                                                         l1_regularization_strength=self.l1_regularization_strength,
                                                         l2_regularization_strength=self.l2_regularization_strength)
            self.AdagradDA = True
        elif self.optimiser in ['GradientDescentOptimizer', tf.train.GradientDescentOptimizer]:
            optimiser_obj = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            raise InputError("Unknown optimiser class. Got %s" % str(self.optimiser))

        return optimiser_obj

    def _set_scoring_function(self, scoring_function):
        """
        This function sets which scoring metrics to use when scoring the results.

        :param scoring_function: name of the scoring function to use
        :type scoring_function: string
        :return: None
        """
        if not is_string(scoring_function):
            raise InputError("Expected a string for variable 'scoring_function'. Got %s" % str(scoring_function))
        if scoring_function.lower() not in ['mae', 'rmse', 'r2']:
            raise InputError("Available scoring functions are 'mae', 'rmse', 'r2'. Got %s" % str(scoring_function))

        self.scoring_function = scoring_function

    def _set_hidden_layers_sizes(self, hidden_layer_sizes):
        """
        This function sets the number of hidden layers and the number of neurons in each hidden layer. The length of the
        tuple tells the number of hidden layers (n_hidden_layers) while each element of the tuple specifies the number
        of hidden neurons in that layer.

        :param hidden_layer_sizes: number of hidden layers and hidden neurons.
        :type hidden_layer_sizes: tuple of length n_hidden_layer
        :return: None
        """
        try:
            iterator = iter(hidden_layer_sizes)
        except TypeError:
            raise InputError("'hidden_layer_sizes' must be a tuple of positive integers. Got a non-iterable object.")

        if None in hidden_layer_sizes:
            raise InputError("'hidden_layer_sizes' must be a tuple of positive integers. Got None elements")
        if not is_positive_integer_array(hidden_layer_sizes):
            raise InputError("'hidden_layer_sizes' must be a tuple of positive integers")

        self.hidden_layer_sizes = np.asarray(hidden_layer_sizes, dtype = int)

    def _set_tensorboard(self, tensorboard, store_frequency, tensorboard_subdir):
        """
        This function prepares all the things needed to use tensorboard when training the estimator.

        :param tensorboard: whether to use tensorboard or not
        :type tensorboard: boolean
        :param store_frequency: Every how many steps to save data to tensorboard
        :type store_frequency: int
        :param tensorboard_subdir: directory where to save the tensorboard data
        :type tensorboard_subdir: string
        :return: None
        """

        if not is_bool(tensorboard):
            raise InputError("Expected boolean value for variable tensorboard. Got %s" % str(tensorboard))
        self.tensorboard = bool(tensorboard)

        if not self.tensorboard:
            return

        if not is_string(tensorboard_subdir):
            raise InputError('Expected string value for variable tensorboard_subdir. Got %s' % str(tensorboard_subdir))

        # TensorBoardLogger will handle all tensorboard related things
        self.tensorboard_logger = TensorBoardLogger(tensorboard_subdir)
        self.tensorboard_subdir = tensorboard_subdir

        if not is_positive_integer(store_frequency):
            raise InputError("Expected positive integer value for variable store_frequency. Got %s" % str(store_frequency))

        if store_frequency > self.iterations:
            print("Only storing final iteration for tensorboard")
            self.tensorboard_logger.set_store_frequency(self.iterations)
        else:
            self.tensorboard_logger.set_store_frequency(store_frequency)

    def _init_weight(self, n1, n2, name):
        """
        This function generates a the matrix of weights to go from layer l to the next layer l+1. It initialises the
        weights from a truncated normal distribution where the standard deviation is 1/sqrt(n2) and the mean is zero.

        :param n1: size of the layer l+1
        :type n1: int
        :param n2: size of the layer l
        :type n2: int
        :param name: name to give to the tensor of weights to make tensorboard clear
        :type name: string

        :return: weights to go from layer l to layer l+1
        :rtype: tf tensor of shape (n1, n2)
        """

        w = tf.Variable(tf.truncated_normal([n1,n2], stddev = 1.0 / np.sqrt(n2), dtype = self.tf_dtype),
                dtype = self.tf_dtype, name = name)

        return w

    def _init_bias(self, n, name):
        """
        This function initialises the biases to go from layer l to layer l+1.

        :param n: size of the layer l+1
        :type n: int
        :param name: name to give to the tensor of biases to make tensorboard clear
        :type name: string
        :return: biases
        :rtype: tf tensor of shape (n, 1)
        """

        b = tf.Variable(tf.zeros([n], dtype = self.tf_dtype), name=name, dtype = self.tf_dtype)

        return b

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

    def _l2_loss(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list of tf tensors
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l2_loss")

        for i in range(self.hidden_layer_sizes.size):
            reg_term += tf.reduce_sum(tf.square(weights[i]))

        return self.l2_reg * reg_term

    def _l1_loss(self, weights):
        """
        Creates the expression for L1-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list of tf tensors
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l1_loss")

        for i in range(self.hidden_layer_sizes.size):
            reg_term += tf.reduce_sum(tf.abs(weights[i]))

        return self.l1_reg * reg_term

    def _get_batch_size(self):
        """
        Determines the actual batch size. If set to auto, the batch size will be set to 100.
        If the batch size is larger than the number of samples, it is truncated and a warning
        is printed.

        Furthermore the returned batch size will be slightly modified from the user input if
        the last batch would be tiny compared to the rest.

        :return: Batch size
        :rtype: int
        """

        if self.batch_size == 'auto':
            batch_size = min(100, self.n_samples)
        else:
            if self.batch_size > self.n_samples:
                print("Warning: batch_size larger than sample size. It is going to be clipped")
                return min(self.n_samples, self.batch_size)
            else:
                batch_size = self.batch_size

        # see if the batch size can be modified slightly to make sure the last batch is similar in size
        # to the rest of the batches
        # This is always less that the requested batch size, so no memory issues should arise
        better_batch_size = ceil(self.n_samples, ceil(self.n_samples, batch_size))

        return better_batch_size

    def plot_cost(self, filename = None):
        """
        Plots the value of the cost function as a function of the iterations.

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
        df["Iterations"] = range(len(self.training_cost))
        df["Training cost"] = self.training_cost
        f = sns.lmplot('Iterations', 'Training cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)
        f.set(yscale = "log")

        if filename == None:
            plt.show()
        elif is_string(filename):
            plt.savefig(filename)
        else:
            raise InputError("Wrong data type of variable 'filename'. Expected string")

    def correlation_plot(self, y_nn, y_true, filename = ''):
        """
        Creates a correlation plot between predictions and true values.

        :param y_predicted: Molecular properties predicted by the network
        :type y_predicted: numpy array of shape (n_samples,)
        :param y_true: True molecular properties
        :type y_true: numpy array of shape (n_samples,)
        :param filename: File to save the plot to. If '' the plot is shown instead of saved.
                         If the dimensionality of y is higher than 1, the filename will be prefixed
                         by the dimension.
        :type filename: string
        :return: None
        """

        try:
            import pandas as pd
            import seaborn as sns
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'pandas'")

        if y_nn.shape != y_true.shape:
            raise InputError("Shape mismatch between predicted and true values. %s and %s" % (str(y_nn.shape), str(y_true.shape)))

        if y_nn.ndim == 1 or y_nn.shape[1] == 1:
            df = pd.DataFrame()
            df["Predictions"] = y_nn.ravel()
            df["True"] = y_true.ravel()
            sns.set()
            lm = sns.lmplot('True', 'Predictions', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
            if filename == '':
                plt.show()
            elif is_string(filename):
                plt.savefig(filename)
            else:
                raise InputError("Wrong data type of variable 'filename'. Expected string")
        else:
            for i in range(y_nn.shape[0]):
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
                    plt.savefig(file_)
                else:
                    raise InputError("Wrong data type of variable 'filename'. Expected string")

    def score(self, *args):
        """
        This function calls the appropriate function to score the model. One needs to pass a descriptor and some
        properties to it.

        For MRMP, the parameters should be:
        :param x: The molecular descriptor
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        For ARMP, the arguments should be:
        :param xzs: List containing the descriptor and the nuclear charges [x, zs]
        :type xzs: List of length 2 containing two numpy arrays [x, zs]
        :type x: array of floats of shape (n_samples, n_atoms, n_features)
        :type zs: array of floats of shape (n_samples, n_atoms)
        :param y: The molecular properties
        :type y: array of shape (n_samples,)

        Both MRMP and ARMP can have:
        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: score
        :rtype: float
        """
        return self._score(*args)

    # TODO test
    def _score(self, *args):
        """
        This function calls the appropriate cost function selected by the user. The parameters required are the same as
        for the score function.

        :return: score
        :rtype: float
        """
        if self.scoring_function == 'mae':
            return self._score_mae(*args)
        if self.scoring_function == 'rmse':
            return self._score_rmse(*args)
        if self.scoring_function == 'r2':
            return self._score_r2(*args)

    def fit(self, x, y):
        """
        This function calls the specific fit method of the child classes.

        :param x: The x part of the data
        :type x: for the MRMP class it should be a numpy array of shape (n_samples, n_features), for ARMP it should be a
        list containinf a numpy array of shape (n_samples, n_atoms, n_features) and a numpy array of shape
        (n_samples, n_atoms)
        :param y: the y part of the data
        :type y: a numpy array of shape (n_samples,)
        :return:
        """

        return self._fit(x, y)

    def predict(self, x):
        """
        This function calls the predict function for either ARMP or MRMP.

        :param x: descriptor
        :rtype x: for MRMP this is a numpy array of shape (n_samples, n_features). For ARMP this is a numpy array of
        shape (n_samples, n_atoms, n_features)

        :return: predictions of the molecular properties.
        :rtype: numpy array of shape (n_samples,)
        """
        predictions = self._predict(x)

        if predictions.ndim > 1 and predictions.shape[1] == 1:
            return predictions.ravel()
        else:
            return predictions

### --------------------- ** Molecular representation - molecular properties network ** --------------------------------

class MRMP(_NN):
    """
    Neural network for either
    1) predicting global properties, such as energies, using molecular representations, or
    2) predicting local properties, such as chemical shieldings, using atomic representations.
    """

    def __init__(self, **kwargs):
        """
        Descriptors is used as input to a single or multi layered feed-forward neural network with a single output.
        This class inherits from the _NN class and all inputs not unique to the NN class is passed to the _NN
        parent.

        """

        super(MRMP, self).__init__(**kwargs)

    #TODO upgrade so that this uses tf.Dataset like the ARMP class
    def _fit(self, x, y):
        """
        This function fits the estimator to the data provided.

        :param x: Molecular descriptor.
        :type x: numpy array of shape (n_samples, n_atoms)
        :param y: Molecular properties
        :type y: numpy array of shape (n_samples, )
        """

        # Check that X and y have correct shape
        x, y = check_X_y(x, y, multi_output = False, y_numeric = True, warn_on_dtype = True)

        # reshape to tensorflow friendly shape
        y = np.atleast_2d(y).T

        # Useful quantities
        self.n_features = x.shape[1]
        self.n_samples = x.shape[0]

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
                self.tensorboard_logger.write_weight_histogram(weights)

        with tf.name_scope("Model"):
            y_pred = self._model(tf_x, weights, biases)

        with tf.name_scope("Cost_func"):
            cost = self.cost(y_pred, tf_y, weights)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost)
        # Initialisation of the variables
        init = tf.global_variables_initializer()

        if self.tensorboard:
            self.tensorboard_logger.initialise()

        # This is the total number of batches in which the training set is divided
        n_batches = ceil(self.n_samples, batch_size)

        self.session = tf.Session()

        # Running the graph
        if self.tensorboard:
            self.tensorboard_logger.set_summary_writer(self.session)

        self.session.run(init)

        indices = np.arange(0,self.n_samples, 1)

        for i in range(self.iterations):
            # This will be used to calculate the average cost per iteration
            avg_cost = 0
            # Learning over the batches of data
            for j in range(n_batches):
                batch_x = x[indices][j * batch_size:(j+1) * batch_size]
                batch_y = y[indices][j * batch_size:(j+1) * batch_size]
                if self.AdagradDA:
                    feed_dict = {tf_x: batch_x, tf_y: batch_y, self.global_step:i}
                    opt, c = self.session.run([optimisation_op, cost], feed_dict=feed_dict)
                else:
                    feed_dict = {tf_x: batch_x, tf_y: batch_y}
                    opt, c = self.session.run([optimisation_op, cost], feed_dict=feed_dict)
                avg_cost += c * batch_x.shape[0] / x.shape[0]

                if self.tensorboard:
                    if i % self.tensorboard_logger.store_frequency == 0:
                        self.tensorboard_logger.write_summary(self.session, feed_dict, i, j)

            self.training_cost.append(avg_cost)

            # Shuffle the dataset at each iteration
            np.random.shuffle(indices)

    def _model(self, x, weights, biases):
        """
        Constructs the molecular neural network.

        :param x: descriptor
        :type x: tf.placeholder of shape (n_samples, n_features)
        :param weights: Weights used in the network.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param biases: Biases used in the network.
        :type biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Output
        :rtype: tf.Variable of size (n_samples, n_targets)
        """

        # Calculate the activation of the first hidden layer
        z = tf.add(tf.matmul(x, tf.transpose(weights[0])), biases[0])
        h = self.activation_function(z)

        # Calculate the activation of the remaining hidden layers
        for i in range(self.hidden_layer_sizes.size - 1):
            z = tf.add(tf.matmul(h, tf.transpose(weights[i + 1])), biases[i + 1])
            h = self.activation_function(z)

        # Calculating the output of the last layer
        z = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1], name="output")

        return z

    # TODO test
    def _score_r2(self, x, y, sample_weight=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: The molecular descriptor
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: R^2
        :rtype: float
        """

        y_pred = self.predict(x)
        r2 = r2_score(y, y_pred, sample_weight = sample_weight)
        return r2

    # TODO test
    def _score_mae(self, x, y, sample_weight=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: The molecular descriptor
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        y_pred = self.predict(x)
        mae = (-1.0)*mean_absolute_error(y, y_pred, sample_weight = sample_weight)
        print("Warning! The mae is multiplied by -1 so that it can be minimised in Osprey!")
        return mae

    # TODO test
    def _score_rmse(self, x, y, sample_weight=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: The molecular descriptor
        :type x: array of shape (n_samples, n_features)
        :param y: The target values for each sample in x.
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        y_pred = self.predict(x)
        rmse = np.sqrt(mean_squared_error(y, y_pred, sample_weight = sample_weight))
        return rmse

    def cost(self, y_pred, y, weights):
        """
        Constructs the cost function

        :param y_pred: Predicted molecular properties
        :type y_pred: tf.Variable of size (n_samples, 1)
        :param y: True molecular properties
        :type y: tf.placeholder of shape (n_samples, 1)
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

    def _predict(self, x):
        """
        Use the trained network to make predictions on the descriptor x. This function has to be overwritten by ARMP
        as it uses the atomic decomposed model.

        :param x: The descriptor
        :type x: numpy array of shape (n_samples, n_features)

        :return: Predictions for the target values corresponding to the samples contained in x.
        :rtype: numpy array of shape (n_samples, )

        """

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        check_array(x, warn_on_dtype = True)

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            model = graph.get_tensor_by_name("Model/output:0")
            y_pred = self.session.run(model, feed_dict = {tf_x : x})
            return y_pred

    def save_nn(self, save_dir="saved_model"):
        """
        This function saves the model to be used for later prediction.

        :param save_dir: name of the directory to create to save the model
        :type save_dir: string
        :return: None
        """
        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            model = graph.get_tensor_by_name("Model/output:0")

        tf.saved_model.simple_save(self.session, export_dir=save_dir,
                                   inputs={"Data/Descriptors:0": tf_x},
                                   outputs={"Model/output:0": model})

    def load_nn(self, save_dir="saved_model"):
        """
        This function reloads a model for predictions.
        :param save_dir: the name of the directory where the model is saved.
        :type save_dir: string
        :return: None
        """

        self.session = tf.Session(graph=tf.get_default_graph())
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], save_dir)

### --------------------- ** Atomic representation - molecular properties network ** -----------------------------------

class ARMP(_NN):
    """
    This class contains neural networks that take in atomic representations and they calculate molecular properties
    such as the energies.
    """

    def __init__(self, **kwargs):
        """
        To see what parameters are required, look at the description of the _NN class init.
        This class inherits from the _NN class and all inputs not unique to the ARMP class are passed to the _NN
        parent.
        """

        super(ARMP, self).__init__(**kwargs)

    def _atomic_model(self, x, hidden_layer_sizes, weights, biases):
        """
        Constructs the atomic part of the network. It calculates the output for all atoms as if they all were the same
        element.

        :param x: Atomic descriptor
        :type x: tf tensor of shape (n_samples, n_atoms, n_features)
        :param weights: Weights used in the network for a particular element.
        :type weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param biases: Biases used in the network for a particular element.
        :type biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Output
        :rtype: tf.Variable of size (n_samples, n_atoms)
        """

        # Calculate the activation of the first hidden layer
        z = tf.add(tf.tensordot(x, tf.transpose(weights[0]), axes=1), biases[0])
        h = self.activation_function(z)

        # Calculate the activation of the remaining hidden layers
        for i in range(hidden_layer_sizes.size - 1):
            z = tf.add(tf.tensordot(h, tf.transpose(weights[i + 1]), axes=1), biases[i + 1])
            h = self.activation_function(z)

        # Calculating the output of the last layer
        z = tf.add(tf.tensordot(h, tf.transpose(weights[-1]), axes=1), biases[-1])

        z_squeezed = tf.squeeze(z, axis=[-1])

        return z_squeezed

    def _model(self, x, zs, element_weights, element_biases):
        """
        This generates the molecular model by combining all the outputs from the atomic networks.

        :param x: Atomic descriptor
        :type x: tf tensor of shape (n_samples, n_atoms, n_features)
        :param zs: Nuclear charges of the systems
        :type zs: tf tensor of shape (n_samples, n_atoms)
        :param element_weights: Element specific weights
        :type element_weights: list of tf.Variables of length hidden_layer_sizes.size + 1
        :param element_biases: Element specific biases
        :type element_biases: list of tf.Variables of length hidden_layer_sizes.size + 1
        :return: Predicted properties for all samples
        :rtype: tf tensor of shape (n_samples, 1)
        """

        atomic_energies = tf.zeros_like(zs)

        for i in range(self.elements.shape[0]):

            # Calculating the output for every atom in all data as if they were all of the same element
            atomic_energies_all = self._atomic_model(x, self.hidden_layer_sizes, element_weights[self.elements[i]],
                                                 element_biases[self.elements[i]])  # (n_samples, n_atoms)

            # Figuring out which atomic energies correspond to the current element.
            current_element = tf.expand_dims(tf.constant(self.elements[i], dtype=tf.int32), axis=0)
            where_element = tf.equal(tf.cast(zs, dtype=tf.int32), current_element)  # (n_samples, n_atoms)

            # Extracting the energies corresponding to the right element
            element_energies = tf.where(where_element, atomic_energies_all, tf.zeros_like(zs))

            # Adding the energies of the current element to the final atomic energies tensor
            atomic_energies = tf.add(atomic_energies, element_energies)

        # Summing the energies of all the atoms
        total_energies = tf.reduce_sum(atomic_energies, axis=-1, name="output", keepdims=True)

        return total_energies

    def cost(self, y_pred, y, weights_dict):
        """
        This function calculates the cost function during the training of the neural network.

        :param y_pred: the neural network predictions
        :type y_pred: tensors of shape (n_samples, 1)
        :param y: the truth values
        :type y: tensors of shape (n_samples, 1)
        :param weights_dict: the dictionary containing all of the weights
        :type weights_dict: dictionary where the key is a nuclear charge and the value is a list of tensors of length
        hidden_layer_sizes.size + 1
        :return: value of the cost function
        :rtype: tf.Variable of size (1,)
        """

        err =  tf.square(tf.subtract(y, y_pred))
        cost_function = tf.reduce_mean(err, name="loss")

        if self.l2_reg > 0:
            l2_loss = 0
            for element in weights_dict:
                l2_loss += self._l2_loss(weights_dict[element])
            cost_function += l2_loss
        if self.l1_reg > 0:
            l1_loss = 0
            for element in weights_dict:
                l1_loss += self._l1_loss(weights_dict[element])
            cost_function += l1_loss

        return cost_function

    def _find_elements(self, zs):
        """
        This function finds the unique atomic numbers in Zs and returns them in a list.

        :param zs: nuclear charges
        :type zs: numpy array of floats of shape (n_samples, n_atoms)
        :return: unique nuclear charges
        :rtype: numpy array of floats of shape (n_elements,)
        """

        # Obtaining the unique atomic numbers (but still includes the dummy atoms)
        elements = np.unique(zs)

        # Removing the dummy
        return np.trim_zeros(elements)

    def _fit(self, xzs, y):
        """
        This class fits the neural network to the data. x corresponds to the descriptors and y to the molecular
        property to predict. zs contains the atomic numbers of all the atoms in the data.

        :param xzs: Atomic descriptor and nuclear charges combined in a list
        :type xzs: list of size two containing two numpy arrays x and zs [x, zs]
        :type x: numpy array of floats of shape (n_samples, n_atoms, n_features)
        :type zs: numpy array of floats of shape (n_samples, n_atoms)
        :param y: molecular properties
        :type y: numpy array of floats of shape (n_samples, 1)
        :return: None
        """

        if len(xzs) != 2:
            raise InputError("xzs should be a list of size 2. Got %s" % (len(xzs)))

        x = xzs[0]
        zs = xzs[1]

        # reshape to tensorflow friendly shape
        y = np.atleast_2d(y).T

        # Obtaining the array of unique elements in all samples
        self.elements = self._find_elements(zs)

        # Useful quantities
        self.n_samples = x.shape[0]
        self.n_atoms = x.shape[1]
        self.n_features = x.shape[2]

        # Set the batch size
        batch_size = self._get_batch_size()

        # This is the total number of batches in which the training set is divided
        n_batches = ceil(self.n_samples, batch_size)

        tf.reset_default_graph()

        # Initial set up of the NN
        with tf.name_scope("Data"):
            x_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, self.n_atoms, self.n_features])
            zs_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, self.n_atoms])
            y_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, 1])

            dataset = tf.data.Dataset.from_tensor_slices((x_ph, zs_ph, y_ph))
            dataset = dataset.shuffle(buffer_size=self.n_samples)
            dataset = dataset.batch(batch_size)
            # batched_dataset = dataset.prefetch(buffer_size=batch_size)

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            tf_x, tf_zs, tf_y = iterator.get_next()
            tf_x = tf.identity(tf_x, name="Descriptors")
            tf_zs = tf.identity(tf_zs, name="Atomic-numbers")
            tf_y = tf.identity(tf_y, name="Properties")

        # Creating dictionaries of the weights and biases for each element
        element_weights = {}
        element_biases = {}

        with tf.name_scope("Weights"):
            for i in range(self.elements.shape[0]):
                weights, biases = self._generate_weights(n_out=1)
                element_weights[self.elements[i]] = weights
                element_biases[self.elements[i]] = biases

                # Log weights for tensorboard
                if self.tensorboard:
                    self.tensorboard_logger.write_weight_histogram(weights)

        with tf.name_scope("Model"):
            molecular_energies = self._model(tf_x, tf_zs, element_weights, element_biases)

        with tf.name_scope("Cost_func"):
            cost = self.cost(molecular_energies, tf_y, element_weights)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost)

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        iterator_init = iterator.make_initializer(dataset)

        if self.tensorboard:
            self.tensorboard_logger.initialise()

        self.session = tf.Session()

        # Running the graph
        if self.tensorboard:
            # self.tensorboard_logger.set_summary_writer(self.session)
            file_writer = tf.summary.FileWriter(logdir=self.tensorboard_subdir, graph=self.session.graph)

        self.session.run(init)
        self.session.run(iterator_init, feed_dict={x_ph:x, zs_ph:zs, y_ph:y})


        for i in range(self.iterations):

            self.session.run(iterator_init, feed_dict={x_ph: x, zs_ph: zs, y_ph: y})
            avg_cost = 0

            for j in range(n_batches):
                if self.AdagradDA:
                    opt, c = self.session.run([optimisation_op, cost])
                else:
                    opt, c = self.session.run([optimisation_op, cost])

                avg_cost += c

            # This seems to run the iterator.get_next() op, which gives problems with end of sequence
            # Hence why I re-initialise the iterator
            self.session.run(iterator_init, feed_dict={x_ph: x, zs_ph: zs, y_ph: y})
            if self.tensorboard:
                if i % self.tensorboard_logger.store_frequency == 0:
                    summary = self.session.run(tf.summary.merge_all())
                    file_writer.add_summary(summary, i)

            self.training_cost.append(avg_cost/n_batches)

    def _predict(self, xzs):
        """
        This function uses the trained model to predict the molecular properties.
        It overrites the _NN _predict function because it uses the atomic decomposed model.

        :param xzs: list containing x (atomic descriptor) and zs (nuclear charges).
        :type xzs: [x, zs]
        :type x: np array of floats of shape (n_samples, n_atoms, n_features)
        :type zs: np array of floats of shape (n_samples, n_atoms)
        :return: predicted molecular properties.
        :rtype: a np array of floats of shape (n_samples,)
        """

        if len(xzs) > 2:
            raise InputError("xzs should be a list of size 2. Got %s" % (len(xzs)))

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            model = graph.get_tensor_by_name("Model/output:0")
            y_pred = self.session.run(model, feed_dict={tf_x: xzs[0], tf_zs:xzs[1]})

        return y_pred

    def _score_r2(self, xzs, y, sample_weight=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param xzs: List containing the descriptor and the nuclear charges [x, zs]
        :type xzs: List of length 2 containing two numpy arrays [x, zs]
        :type x: array of floats of shape (n_samples, n_atoms, n_features)
        :type zs: array of floats of shape (n_samples, n_atoms)
        :param y: The molecular properties
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: R^2
        :rtype: float

        """

        y_pred = self.predict(xzs)
        r2 = r2_score(y, y_pred, sample_weight = sample_weight)
        return r2

    def _score_mae(self, xzs, y, sample_weight=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param xzs: List containing the descriptor and the nuclear charges [x, zs]
        :type xzs: List of length 2 containing two numpy arrays [x, zs]
        :type x: array of floats of shape (n_samples, n_atoms, n_features)
        :type zs: array of floats of shape (n_samples, n_atoms)
        :param y: The molecular properties
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        y_pred = self.predict(xzs)
        return mean_absolute_error(y, y_pred, sample_weight = sample_weight)

    def _score_rmse(self, xzs, y, sample_weight=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param xzs: List containing the descriptor and the nuclear charges [x, zs]
        :type xzs: List of length 2 containing two numpy arrays [x, zs]
        :type x: array of floats of shape (n_samples, n_atoms, n_features)
        :type zs: array of floats of shape (n_samples, n_atoms)
        :param y: The molecular properties
        :type y: array of shape (n_samples,)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        y_pred = self.predict(xzs)
        rmse = np.sqrt(mean_squared_error(y, y_pred, sample_weight = sample_weight))
        return rmse

    def save_nn(self, save_dir="saved_model"):
        """
        This function saves the trained model to be used for later prediction.

        :param save_dir: directory in which to save the model
        :type save_dir: string
        :return: None
        """

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            model = graph.get_tensor_by_name("Model/output:0")

        tf.saved_model.simple_save(self.session, export_dir=save_dir,
                                   inputs={"Data/Descriptors:0": tf_x, "Data/Atomic-numbers:0": tf_zs},
                                   outputs={"Model/output:0": model})

    def load_nn(self, save_dir="saved_model"):
        """
        This function reloads a model for predictions.

        :param save_dir: the name of the directory where the model is saved.
        :type save_dir: string
        :return: None
        """

        self.session = tf.Session(graph=tf.get_default_graph())
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], save_dir)