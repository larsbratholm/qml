# MIT License
#
# Copyright (c) 2018 Silvia Amabilino, Lars Andersen Bratholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module containing the general neural network class and the child classes for the molecular and atomic neural networks.
"""

from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator
from qml.aglaia.symm_funct import generate_parkhill_acsf, generate_parkhill_acsf_single
from qml.utils.utils import InputError, ceil, is_positive_or_zero, is_positive_integer, is_positive, \
        is_bool, is_positive_integer_or_zero, is_string, is_positive_integer_array, is_array_like, is_none, \
        check_global_representation, check_y, check_sizes, check_dy, check_classes, is_numeric_array, is_non_zero_integer, \
    is_positive_integer_or_zero_array, check_local_representation, check_dgdr, check_xyz
from qml.aglaia.tf_utils import TensorBoardLogger, partial_derivatives
from qml.representations import generate_acsf
from qml.aglaia.graceful_killer import GracefulKiller

try:
    from qml.data import Compound
    from qml import representations as qml_rep
except ImportError:
    raise ImportError("The module qml is required")

try:
    import tensorflow
except ImportError:
    raise ImportError("Tensorflow 1.8 is required to run neural networks.")

class _NN(BaseEstimator):

    """
    Parent class for training multi-layered neural networks on molecular or atomic properties via Tensorflow
    """

    def __init__(self, hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                 iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                 activation_function, optimiser, beta1, beta2, epsilon,
                 rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                 l1_regularization_strength,l2_regularization_strength, tensorboard_subdir):

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

        self._set_optimiser_type(optimiser)

        # Placholder variables for data
        self.xyz = None
        self.compounds = None
        self.g = None
        self.properties = None
        self.gradients = None
        self.classes = None
        self.dg_dr = None
        self.elements = None
        self.element_pairs = None

        # To enable restart model
        self.loaded_model = False

        # To enable restart model
        self.loaded_model = False

    def _set_activation_function(self, activation_function):
        """
        This function sets which activation function will be used in the model.

        :param activation_function: name of the activation function to use
        :type activation_function: string or tf class
        :return: None
        """
        if activation_function in ['sigmoid', 'tanh', 'elu', 'softplus', 'softsign', 
                'relu', 'relu6', 'crelu', 'relu_x']:
            self.activation_function = activation_function
        else:
            raise InputError("Unknown activation function. Got %s" % str(activation_function))

    def _update_activation_function(self):
        """
        Has to set activation function at fit time due to sklearn requiring that
        an init param X has the same id as self.X.
        """

        if self.activation_function == 'sigmoid':
            self.activation_function = tf.nn.sigmoid
        elif self.activation_function == 'tanh':
            self.activation_function = tf.nn.tanh
        elif self.activation_function == 'elu':
            self.activation_function = tf.nn.elu
        elif self.activation_function == 'softplus':
            self.activation_function = tf.nn.softplus
        elif self.activation_function == 'softsign':
            self.activation_function = tf.nn.softsign
        elif self.activation_function == 'relu':
            self.activation_function = tf.nn.relu
        elif self.activation_function == 'relu6':
            self.activation_function = tf.nn.relu6
        elif self.activation_function == 'crelu':
            self.activation_function = tf.nn.crelu
        elif self.activation_function == 'relu_x':
            self.activation_function = tf.nn.relu_x

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
            self.batch_size = batch_size
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
        self.learning_rate = learning_rate

    def _set_iterations(self, iterations):
        """
        This function sets the number of iterations that will be carried out by the optimiser.

        :param iterations: number of iterations
        :type l1_reg: int
        :return: None
        """
        if not is_positive_integer(iterations):
            raise InputError("Expected positive integer value for variable iterations. Got %s" % str(iterations))
        self.iterations = iterations

    # TODO check that the estimators actually use this
    def _update_tf_dtype(self):
        """
        Has to set dtype at fit time due to sklearn requiring that
        an init param X has the same id as self.X.
        """
        # 2 == tf.float64 and 1 == tf.float32 for some reason
        # np.float64 recognised as tf.float64 as well
        if self.tf_dtype in ['64', 64, 'float64', tf.float64]:
            self.tf_dtype = tf.float64
        elif self.tf_dtype in ['32', 32, 'float32', tf.float32]:
            self.tf_dtype = tf.float32
        elif self.tf_dtype in ['16', 16, 'float16', tf.float16]:
            self.tf_dtype = tf.float16
        else:
            raise InputError("Unknown tensorflow data type. Got %s" % str(tf_dtype))

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
        if tf_dtype in ['64', 64, 'float64', tf.float64, '32', 32, 'float32', tf.float32,
                '16', 16, 'float16', tf.float16]:
            self.tf_dtype = tf_dtype
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
        self.beta1 = beta1
        self.beta2 = beta2

        if not is_positive(epsilon):
            raise InputError("Expected positive float value for variable epsilon. Got %s" % str(epsilon))
        self.epsilon = epsilon

        if not is_positive(rho):
            raise InputError("Expected positive float value for variable rho. Got %s" % str(rho))
        self.rho = rho

        if not is_positive(initial_accumulator_value) and not is_positive(initial_gradient_squared_accumulator_value):
            raise InputError("Expected positive float value for accumulator values. Got %s and %s" %
                             (str(initial_accumulator_value), str(initial_gradient_squared_accumulator_value)))
        self.initial_accumulator_value = initial_accumulator_value
        self.initial_gradient_squared_accumulator_value = initial_gradient_squared_accumulator_value

        if not is_positive_or_zero(l1_regularization_strength) and not is_positive_or_zero(l2_regularization_strength):
            raise InputError("Expected positive or zero float value for regularisation variables. Got %s and %s" %
                             (str(l1_regularization_strength), str(l2_regularization_strength)))
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength

    def _set_optimiser_type(self, optimiser):
        """
        This function sets which numerical optimisation algorithm will be used for training.

        :param optimiser: Optimiser
        :type optimiser: string or tf class
        :return: tf optimiser to use
        :rtype: tf class
        """
        self.AdagradDA = False
        if optimiser in ['AdamOptimizer', tf.train.AdamOptimizer, 'AdadeltaOptimizer', tf.train.AdadeltaOptimizer,
                'AdagradOptimizer', tf.train.AdagradOptimizer, 'AdagradDAOptimizer', tf.train.AdagradDAOptimizer,
                'GradientDescentOptimizer', tf.train.GradientDescentOptimizer]:
            self.optimiser = optimiser
        else:
            raise InputError("Unknown optimiser. Got %s" % str(optimiser))

    def _set_optimiser(self):
        """
        This function instantiates an object from the optimiser class that has been selected by the user. It also sets
        the parameters for the optimiser.

        :return: Optimiser with set parameters
        :rtype: object of tf optimiser class
        """
        self.AdagradDA = False
        if self.optimiser in ['AdamOptimizer', tf.train.AdamOptimizer]:
            optimiser_obj = tf.train.AdamOptimizer(learning_rate=float(self.learning_rate), beta1=float(self.beta1), beta2=float(self.beta2),
                                                    epsilon=float(self.epsilon))
        elif self.optimiser in ['AdadeltaOptimizer', tf.train.AdadeltaOptimizer]:
             optimiser_obj = tf.train.AdadeltaOptimizer(learning_rate=float(self.learning_rate), rho=float(self.rho), epsilon=float(self.epsilon))
        elif self.optimiser in ['AdagradOptimizer', tf.train.AdagradOptimizer]:
             optimiser_obj = tf.train.AdagradOptimizer(learning_rate=float(self.learning_rate),
                                                       initial_accumulator_value=float(self.initial_accumulator_value))
        elif self.optimiser in ['AdagradDAOptimizer', tf.train.AdagradDAOptimizer]:
            self.global_step = tf.placeholder(dtype=tf.int64)
            optimiser_obj = tf.train.AdagradDAOptimizer(learning_rate=float(self.learning_rate), global_step=self.global_step,
                                                         initial_gradient_squared_accumulator_value=float(self.initial_gradient_squared_accumulator_value),
                                                         l1_regularization_strength=float(self.l1_regularization_strength),
                                                         l2_regularization_strength=float(self.l2_regularization_strength))
            self.AdagradDA = True
        elif self.optimiser in ['GradientDescentOptimizer', tf.train.GradientDescentOptimizer]:
            optimiser_obj = tf.train.GradientDescentOptimizer(learning_rate=float(self.learning_rate))
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
        if scoring_function.lower() not in ['mae', 'rmse', 'r2', 'negmae']:
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
        self.tensorboard = tensorboard

        if not self.tensorboard:
            return

        if not is_string(tensorboard_subdir):
            raise InputError('Expected string value for variable tensorboard_subdir. Got %s' % str(tensorboard_subdir))

        # This line is needed for when the estimator is cloned
        self.tensorboard_subdir = tensorboard_subdir
        self.store_frequency = store_frequency

        # TensorBoardLogger will handle all tensorboard related things
        self.tensorboard_logger_training = TensorBoardLogger(tensorboard_subdir + '/training')
        self.tensorboard_subdir_training = tensorboard_subdir + '/training'

        self.tensorboard_logger_representation = TensorBoardLogger(tensorboard_subdir + '/representation')
        self.tensorboard_subdir_representation = tensorboard_subdir + '/representation'

        if not is_positive_integer(store_frequency):
            raise InputError("Expected positive integer value for variable store_frequency. Got %s" % str(store_frequency))

        if store_frequency > self.iterations:
            print("Only storing final iteration for tensorboard")
            self.tensorboard_logger_training.set_store_frequency(self.iterations)
        else:
            self.tensorboard_logger_training.set_store_frequency(store_frequency)

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
        the number of features in the representation and the number of outputs. The weights are initialised from
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

    def _set_slatm_parameters(self, params):
        """
        This function sets the parameters for the slatm representation.
        :param params: dictionary
        :return: None
        """

        self.representation_params = {'slatm_sigma1': 0.05, 'slatm_sigma2': 0.05, 'slatm_dgrid1': 0.03, 'slatm_dgrid2': 0.03,
                                 'slatm_rcut': 4.8, 'slatm_rpower': 6, 'slatm_alchemy': False}

        if params is not None:
            for key, value in params.items():
                if key in self.representation_params:
                    self.representation_params[key] = value

            self._check_slatm_values()

    def _set_acsf_parameters(self, params):
        """
        This function sets the parameters for the acsf representation.
        :param params: dictionary
        :return: None
        """

        self.acsf_parameters = {'rcut': 5.0, 'acut': 5.0, 'nRs2': 5, 'nRs3': 5, 'nTs': 5,
                                      'zeta': 220.127, 'eta': 30.8065}

        if params is not None:
            for key, value in params.items():
                if key in self.acsf_parameters:
                    self.acsf_parameters[key] = value

            self._check_acsf_values()

    def score(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        This function calls the appropriate function to score the model. One needs to pass a representation and some
        properties to it or alternatively if the compounds/representations and the properties are stored in the class one
        can pass indices.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients of the properties or none
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3) or None
        :param classes: either the classes to do the NN decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates or None
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: score
        :rtype: float
        """
        return self._score(x, y, classes, dy, dgdr)

    def _score(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        This function calls the appropriate function to score the model. One needs to pass a representation and some
        properties to it or alternatively if the compounds/representations and the properties are stored in the class one
        can pass indices.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients of the properties or none
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3) or None
        :param classes: either the classes to do the NN decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates or None
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: score
        :rtype: float
        """
        if self.scoring_function == 'mae':
            return self._score_mae(x, y, classes, dy, dgdr)
        elif self.scoring_function == 'negmae':
            return -self._score_mae(x, y, classes, dy, dgdr)
        elif self.scoring_function == 'rmse':
            return self._score_rmse(x, y, classes, dy, dgdr)
        elif self.scoring_function == 'r2':
            return self._score_r2(x, y, classes, dy, dgdr)

    def generate_compounds(self, filenames):
        """
        Creates QML compounds. Needs to be called before fitting.

        :param filenames: path of xyz-files
        :type filenames: list
        """

        # Check that the number of properties match the number of compounds if the properties have already been set
        if self.properties is None:
            pass
        else:
            if self.properties.size == len(filenames):
                pass
            else:
                raise InputError("Number of properties (%d) does not match number of compounds (%d)"
                                 % (self.properties.size, len(filenames)))

        self.compounds = np.empty(len(filenames), dtype=object)
        for i, filename in enumerate(filenames):
            self.compounds[i] = Compound(filename)

    def generate_compounds_from_xyz(self, xyz, zs):
        """
        Creates QML compounds. It initialises them from xyz and zs arrays rather than *.xyz files.
        :param xyz: cartesian coordintes
        :type xyz: numpy array of shape (n_atoms, 3)
        :param zs: nuclear charges
        :type zs: numpy array of shape (n_atoms, )
        :return: None
        """

        self.compounds = np.empty(xyz.shape[0], dtype=object)
        elements = {6:'C', 7:'N', 1:'H'}

        for i in range(xyz.shape[0]):
            a_compound = Compound()
            a_compound.use_xyz_zs_array(xyz[i], zs[i])
            self.compounds[i] = a_compound

    def generate_representation(self, xyz=None, classes=None, method="fortran"):
        """
        This function can generate representations either from the data contained in the compounds or from xyz data passed
        through the argument. If the Compounds have already being set and xyz data is given, it complains.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: The classes to do the atomic decomposition of the networks (most commonly nuclear charges)
        :type classes: numpy array of shape (n_samples, n_atoms)
        :param method: for ARMP_G there are 2 ways of generating the descriptor, one with fortran and one with tf. This flag enables to chose
        :type method: string
        :return: None
        """

        # TODO need to add a way of just setting XYZ
        if is_none(self.compounds) and is_none(xyz) and is_none(classes):
            raise InputError("QML compounds need to be created in advance or Cartesian coordinates need to be passed in "
                             "order to generate the representation.")

        if not is_none(self.g):
            raise InputError("The representations have already been set!")

        if self.compounds is None:

            self.g, self.classes = self._generate_representations_from_data(xyz, classes, method)

        elif is_none(xyz):
            # Make representations from compounds

            self.g, self.classes = self._generate_representations_from_compounds()
        else:
            raise InputError("Compounds have already been set but new xyz data is being passed.")

    def set_properties(self, properties):
        """
        Set properties. Needed to be called before fitting.

        :param y: array of properties of size (nsamples,)
        :type y: array
        """
        if properties is None:
            raise InputError("Properties cannot be set to none.")
        else:
            if is_numeric_array(properties) and np.asarray(properties).ndim == 1:
                self.properties = np.asarray(properties)
            else:
                raise InputError(
                    'Variable "properties" expected to be array like of dimension 1. Got %s' % str(properties))

    def set_xyz(self, xyz):

        if is_none(xyz):
            raise InputError("The cartesian coordinates cannot be set to none.")
        else:
            if is_numeric_array(xyz) and np.asarray(xyz).ndim == 3:
                self.xyz = np.asarray(xyz)
            else:
                raise InputError(
                    'Variable "xyz" expected to be array like of dimension 3. Got %s' % str(xyz.shape))

    def set_representations(self, representations):
        """
        This function takes representations as input and stores them inside the class.

        :param representations: global or local representations
        :type representations: numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features)
        """

        if not is_none(self.g):
            raise InputError("The representations have already been set!")

        if is_none(representations):
            raise InputError("Descriptor cannot be set to none.")
        else:
            if is_numeric_array(representations):
                self._set_representation(representations)
            else:
                raise InputError('Variable "representation" expected to be array like.')

    def set_gradients(self, gradients):
        """
        This function enables to set the gradient information.

        :param gradients: The gradients of the properties with respect to the input. For example, forces.
        :type gradients: numpy array (for example, numpy array of shape (n_samples, n_atoms, 3))
        :return: None
        """

        if gradients is None:
            raise InputError("Gradients cannot be set to none.")
        else:
            if is_numeric_array(gradients):
                if len(np.asarray(gradients).shape) != 3 or np.asarray(gradients).shape[-1] != 3:
                    raise InputError("The gradients should be a three dimensional array with the last dimension equal to 3.")
                self.gradients = np.asarray(gradients)
            else:
                raise InputError('Variable "gradients" expected to be array like.')

    def set_classes(self, classes):
        """
        This function stores the classes to be used during training for local networks.

        :param classes: what class does each atom belong to.
        :type classes: numpy array of shape (n_samples, n_atoms) of ints
        :return: None
        """
        if classes is None:
            raise InputError("Classes cannot be set to none.")
        else:
            if is_positive_integer_or_zero_array(classes):
                self.classes = np.asarray(classes)
            else:
                raise InputError('Variable "classes" expected to be array like of positive integers.')


    def fit(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        This function calls the specific fit method of the child classes.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param classes: either the classes to do the NN decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dy: either the gradients of the properties or none
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates or None
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: None
        """

        # Sets activation_function and tf_dtypes at fit time for sklearn compatibility
        self._update_activation_function()
        self._update_tf_dtype()

        return self._fit(x, y, classes, dy, dgdr)

    def _check_slatm_values(self):
        """
        This function checks that the parameters passed to slatm make sense.
        :return: None
        """
        if not is_positive(self.representation_params['slatm_sigma1']):
            raise InputError("Expected positive float for variable 'slatm_sigma1'. Got %s." % str(self.representation_params['slatm_sigma1']))

        if not is_positive(self.representation_params['slatm_sigma2']):
            raise InputError("Expected positive float for variable 'slatm_sigma2'. Got %s." % str(self.representation_params['slatm_sigma2']))

        if not is_positive(self.representation_params['slatm_dgrid1']):
            raise InputError("Expected positive float for variable 'slatm_dgrid1'. Got %s." % str(self.representation_params['slatm_dgrid1']))

        if not is_positive(self.representation_params['slatm_dgrid2']):
            raise InputError("Expected positive float for variable 'slatm_dgrid2'. Got %s." % str(self.representation_params['slatm_dgrid2']))

        if not is_positive(self.representation_params['slatm_rcut']):
            raise InputError("Expected positive float for variable 'slatm_rcut'. Got %s." % str(self.representation_params['slatm_rcut']))

        if not is_non_zero_integer(self.representation_params['slatm_rpower']):
            raise InputError("Expected non-zero integer for variable 'slatm_rpower'. Got %s." % str(self.representation_params['slatm_rpower']))

        if not is_bool(self.representation_params['slatm_alchemy']):
            raise InputError("Expected boolean value for variable 'slatm_alchemy'. Got %s." % str(self.representation_params['slatm_alchemy']))

    def _check_acsf_values(self):
        """
        This function checks that the user input parameters to acsf make sense.
        :return: None
        """

        if not is_positive(self.acsf_parameters['rcut']):
            raise InputError(
                "Expected positive float for variable 'rcut'. Got %s." % str(self.acsf_parameters['rcut']))

        if not is_positive(self.acsf_parameters['acut']):
            raise InputError(
                "Expected positive float for variable 'acut'. Got %s." % str(self.acsf_parameters['acut']))

        if not is_positive_integer(self.acsf_parameters['nRs2']):
            raise InputError("Expected positinve integer for 'nRs2. Got %s." % (self.acsf_parameters['nRs2']))

        if not is_positive_integer(self.acsf_parameters['nRs3']):
            raise InputError("Expected positinve integer for 'nRs3. Got %s." % (self.acsf_parameters['nRs3']))

        if not is_positive_integer(self.acsf_parameters['nTs']):
            raise InputError("Expected positinve integer for 'nTs. Got %s." % (self.acsf_parameters['nTs']))

        if is_numeric_array(self.acsf_parameters['eta']) or is_numeric_array(self.acsf_parameters['eta']):
            raise InputError("Expecting a scalar value for eta parameters.")

        if is_numeric_array(self.acsf_parameters['zeta']):
            raise InputError("Expecting a scalar value for zeta. Got %s." % (self.acsf_parameters['zeta']))

    def _get_msize(self, pad = 0):
        """
        Gets the maximum number of atoms in a single molecule. To support larger molecules
        an optional padding can be added by the ``pad`` variable.

        :param pad: Add an integer padding to the returned dictionary
        :type pad: integer

        :return: largest molecule with respect to number of atoms.
        :rtype: integer

        """

        if self.compounds.size == 0:
            raise RuntimeError("QML compounds have not been generated")
        if not is_positive_integer_or_zero(pad):
            raise InputError("Expected variable 'pad' to be a positive integer or zero. Got %s" % str(pad))

        nmax = max(mol.natoms for mol in self.compounds)

        return nmax + pad

    def _get_asize(self, pad = 0):
        """
        Gets the maximum occurrences of each element in a single molecule. To support larger molecules
        an optional padding can be added by the ``pad`` variable.

        :param pad: Add an integer padding to the returned dictionary
        :type pad: integer

        :return: dictionary of the maximum number of occurences of each element in a single molecule.
        :rtype: dictionary

        """

        if self.compounds.size == 0:
            raise RuntimeError("QML compounds have not been generated")
        if not is_positive_integer_or_zero(pad):
            raise InputError("Expected variable 'pad' to be a positive integer or zero. Got %s" % str(pad))

        asize = {}

        for mol in self.compounds:
            for key, value in mol.natypes.items():
                if key not in asize:
                    asize[key] = value + pad
                    continue
                asize[key] = max(asize[key], value + pad)

        return asize

    def _get_slatm_mbtypes(self, arr):
        """
        This function takes an array containing all the classes that are present in a data set and returns a list of all
        the unique classes present, all the possible pairs and triplets of classes.

        :param arr: classes for each atom in a data set
        :type arr: numpy array of shape (n_samples, n_atoms)
        :return: unique single, pair and triplets of classes
        :rtype: list of lists
        """

        return qml_rep.get_slatm_mbtypes(arr)

    def _get_xyz_from_compounds(self, indices):
        """
        This function takes some indices and returns the xyz of the compounds corresponding to those indices.

        :param indices: indices of the compounds to use for training
        :type indices: numpy array of ints of shape (n_samples, )
        :return: the xyz of the specified compounds
        :rtype: numpy array of shape (n_samples, n_atoms, 3)
        """

        xyzs = []
        zs = []
        max_n_atoms = 0

        for compound in self.compounds[indices]:
            xyzs.append(compound.coordinates)
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            xyz_padding = np.zeros((missing_n_atoms, 3))
            xyzs[i] = np.concatenate((xyzs[i], xyz_padding))

        xyzs = np.asarray(xyzs, dtype=np.float32)

        return xyzs

    def _get_properties(self, indices):
        """
        This returns the properties that have been set through QML.

        :param indices: The indices of the properties to return
        :type indices: numpy array of ints of shape (n_samples, )
        :return: the properties of the compounds specified
        :rtype: numpy array of shape (n_samples, 1)
        """

        return np.atleast_2d(self.properties[indices]).T

    def _get_classes_from_compounds(self, indices):
        """
        This returns the classes that have been set through QML.

        :param indices: The indices of the properties to return
        :type indices: numpy array of ints of shape (n_samples, )
        :return: classes of the compounds specified
        :rtype: numpy array of shape (n_samples, n_atoms)
        """

        zs = []
        max_n_atoms = 0

        for compound in self.compounds[indices]:
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            zs_padding = np.zeros(missing_n_atoms)
            zs[i] = np.concatenate((zs[i], zs_padding))

        return np.asarray(zs, dtype=np.float32)

    def predict(self, x, classes=None, dgdr=None):
        """
        This function calls the predict function for either ARMP or MRMP.

        :param x: representation or indices
        :type x: numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or an array of ints
        :param classes: the classes to use for atomic decomposition
        :type classes: numpy array of shape (n_sample, n_atoms)
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates or None
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)


        :return: predictions of the molecular properties.
        :rtype: numpy array of shape (n_samples,)
        """
        predictions = self._predict(x, classes, dgdr)

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

    def __init__(self, hidden_layer_sizes=(5,), l1_reg=0.0, l2_reg=0.0001, batch_size='auto', learning_rate=0.001,
                 iterations=500, tensorboard=False, store_frequency=200, tf_dtype=tf.float32, scoring_function='mae',
                 activation_function="sigmoid", optimiser=tf.train.AdamOptimizer, beta1=0.9, beta2=0.999, epsilon=1e-08,
                 rho=0.95, initial_accumulator_value=0.1, initial_gradient_squared_accumulator_value=0.1,
                 l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                 tensorboard_subdir=os.getcwd() + '/tensorboard', representation_name='unsorted_coulomb_matrix', representation_params=None):
        """
        Descriptors is used as input to a single or multi layered feed-forward neural network with a single output.
        This class inherits from the _NN class and all inputs not unique to the NN class is passed to the _NN
        parent.

        """

        super(MRMP, self).__init__(hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                 iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                 activation_function, optimiser, beta1, beta2, epsilon,
                 rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                 l1_regularization_strength,l2_regularization_strength, tensorboard_subdir)

        self._initialise_representation(representation_name, representation_params)

    def _initialise_representation(self, representation, parameters):
        """
        This function sets the representation and the parameters of the representation.

        :param representation: the name of the representation
        :type representation: string
        :param parameters: all the parameters of the representation.
        :type parameters: dictionary
        :return: None
        """

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation_name = representation.lower()

        if parameters is not None:
            if not type(parameters) is dict:
                raise InputError("The representation parameters passed should be either None or a dictionary.")

        if self.representation_name == 'slatm':

            self._set_slatm_parameters(parameters)

        else:

            if not is_none(parameters):
                raise InputError("The representation %s does not take any additional parameters." % (self.representation_name))

    def _set_representation(self, representation):
        """
        This function takes representations as input and stores them inside the class.

        :param representations: global representations
        :type representations: numpy array of shape (n_samples, n_features)
        return: None
        """

        if len(representation.shape) != 2:
            raise InputError("The representation should have a shape (n_samples, n_features). Got %s" % (str(representation.shape)))

        self.g = representation

    def _generate_representations_from_data(self, xyz, classes, method):
        """
        This function makes the representation from xyz data and nuclear charges.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: classes for atomic decomposition
        :type classes: None
        :param method: What method to use
        :type method: string
        :return: numpy array of shape (n_samples, n_features) and None
        """
        # TODO implement
        raise InputError("Not implemented yet. Use compounds.")

    def _generate_representations_from_compounds(self):
        """
        This function generates the representations from the compounds.

        :return: the representation and None (in the ARMP class this would be the classes for atomic decomposition)
        :rtype: numpy array of shape (n_samples, n_features) and None
        """

        if self.compounds is None:
            raise InputError("This should never happen.")

        n_samples = len(self.compounds)

        if self.representation_name == 'unsorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((n_samples, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds):
                mol.generate_coulomb_matrix(size = nmax, sorting = "unsorted")
                x[i] = mol.representation

        elif self.representation_name == 'sorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((n_samples, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds):
                mol.generate_coulomb_matrix(size = nmax, sorting = "row-norm")
                x[i] = mol.representation

        elif self.representation_name == "bag_of_bonds":
            asize = self._get_asize()
            x = np.empty(n_samples, dtype=object)
            for i, mol in enumerate(self.compounds):
                mol.generate_bob(asize = asize)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        elif self.representation_name == "slatm":
            mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
            x = np.empty(n_samples, dtype=object)
            for i, mol in enumerate(self.compounds):
                mol.generate_slatm(mbtypes, local = False, sigmas = [self.representation_params['slatm_sigma1'],
                                                                     self.representation_params['slatm_sigma2']],
                                   dgrids = [self.representation_params['slatm_dgrid1'], self.representation_params['slatm_dgrid2']],
                                   rcut = self.representation_params['slatm_rcut'],
                                   alchemy = self.representation_params['slatm_alchemy'],
                                   rpower = self.representation_params['slatm_rpower'])
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        else:

            raise InputError("This should never happen. Unrecognised representation. Got %s." % str(self.representation))

        return x, None

    #TODO upgrade so that this uses tf.Dataset like the ARMP class
    def _fit(self, x, y, classes, dy, dgdr):
        """
        This function fits a NON atomic decomposed network to the data.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param classes: None
        :type classes: None
        :param dy: None
        :type dy: None
        :param dg_dr: None
        :type dg_dr: None

        :return: None
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        # Useful quantities
        self.n_features = x_approved.shape[1]
        self.n_samples = x_approved.shape[0]

        # Set the batch size
        batch_size = self._get_batch_size()

        tf.reset_default_graph()

        # Initial set up of the NN
        with tf.name_scope("Data"):
            tf_x = tf.placeholder(self.tf_dtype, [None, self.n_features], name="Descriptors")
            tf_y = tf.placeholder(self.tf_dtype, [None, 1], name="Properties")

        # Either initialise the weights and biases or restart training from wherever it was stopped
        with tf.name_scope("Weights"):
            weights, biases = self._generate_weights(n_out = 1)

            # Log weights for tensorboard
            if self.tensorboard:
                self.tensorboard_logger_training.write_weight_histogram(weights)

        with tf.name_scope("Model"):
            y_pred = self._model(tf_x, weights, biases)

        with tf.name_scope("Cost_func"):
            cost = self._cost(y_pred, tf_y, weights)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost)
        # Initialisation of the variables
        init = tf.global_variables_initializer()

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()

        # This is the total number of batches in which the training set is divided
        n_batches = ceil(self.n_samples, batch_size)

        self.session = tf.Session()

        # Running the graph
        if self.tensorboard:
            self.tensorboard_logger_training.set_summary_writer(self.session)

        self.session.run(init)

        indices = np.arange(0,self.n_samples, 1)

        for i in range(self.iterations):
            # This will be used to calculate the average cost per iteration
            avg_cost = 0
            # Learning over the batches of data
            for j in range(n_batches):
                batch_x = x_approved[indices][j * batch_size:(j + 1) * batch_size]
                batch_y = y_approved[indices][j * batch_size:(j+1) * batch_size]
                if self.AdagradDA:
                    feed_dict = {tf_x: batch_x, tf_y: batch_y, self.global_step:i}
                    opt, c = self.session.run([optimisation_op, cost], feed_dict=feed_dict)
                else:
                    feed_dict = {tf_x: batch_x, tf_y: batch_y}
                    opt, c = self.session.run([optimisation_op, cost], feed_dict=feed_dict)
                avg_cost += c * batch_x.shape[0] / x_approved.shape[0]

                if self.tensorboard:
                    if i % self.tensorboard_logger_training.store_frequency == 0:
                        self.tensorboard_logger_training.write_summary(self.session, feed_dict, i, j)

            self.training_cost.append(avg_cost)

            # Shuffle the dataset at each iteration
            np.random.shuffle(indices)

    def _model(self, x, weights, biases):
        """
        Constructs the molecular neural network.

        :param x: representation
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

    def _score_r2(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None
        :param dg_dr: None
        :type dg_dr: None

        :return: R^2
        :rtype: float
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        y_pred = self.predict(x_approved)
        r2 = r2_score(y_approved, y_pred, sample_weight = None)
        return r2

    def _score_mae(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None
        :param dg_dr: None
        :type dg_dr: None

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        y_pred = self.predict(x_approved)
        mae = (-1.0)*mean_absolute_error(y_approved, y_pred, sample_weight = None)
        print("Warning! The mae is multiplied by -1 so that it can be minimised in Osprey!")
        return mae

    def _score_rmse(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None
        :param dg_dr: None
        :type dg_dr: None

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        y_pred = self.predict(x_approved)
        rmse = np.sqrt(mean_squared_error(y_approved, y_pred, sample_weight = None))
        return rmse

    def _check_inputs(self, x, y, classes, dy, dgdr):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_atoms, 3)
        :param y: None or energies
        :type y: None or numpy array of floats of shape (n_samples,)
        :param dy: None
        :type dy: None
        :param classes: None
        :type classes: None
        :param dg_dr: None
        :type dg_dr: None

        :return: the approved x, y dy and classes
        :rtype: numpy array of shape (n_samples, n_features), (n_samples, 1), None, None
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if not is_none(dy) and not is_none(classes):

            raise InputError("MRMP estimator cannot predict gradients and do atomic decomposition.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.g):
                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.g, _ = self._generate_representations_from_compounds()
            if is_none(self.properties):

                raise InputError("The properties need to be set in advance.")

            approved_x = self.g[x]
            approved_y = self._get_properties(x)
            approved_dy = None
            approved_classes = None

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        else:

            if is_none(y):
                raise InputError("y cannot be of None type.")

            approved_x = check_global_representation(x)
            approved_y = check_y(y)
            approved_dy = None
            approved_classes = None

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        return approved_x, approved_y, approved_dy, approved_classes

    def _check_predict_input(self, x, classes, dgdr):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_features)
        :param classes: None
        :type classes: None
        :param dg_dr: None
        :type dg_dr: None

        :return: the approved x and classes
        :rtype: numpy array of shape (n_samples, n_features), None
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if classes is not None:
            raise InputError("MRMP estimator cannot do atomic decomposition.")

        if not is_none(dgdr):
            raise InputError("MRMP does not need gradients of the representation.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.g):
                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.g, _ = self._generate_representations_from_compounds()
            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")

            approved_x = self.g[x]
            approved_classes = None

        else:

            approved_x = check_global_representation(x)
            approved_classes = None

        return approved_x, approved_classes

    def _cost(self, y_pred, y, weights):
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
        if self.l2_reg >= 0:
            l2_loss = self._l2_loss(weights)
            cost = cost + l2_loss
        if self.l1_reg >= 0:
            l1_loss = self._l1_loss(weights)
            cost = cost + l1_loss

        return cost

    def _predict(self, x, classes, dgdr):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments. Then, the data is
        used as input to the trained network to predict some properties.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_features)
        :param classes: None
        :type classes: None
        :param dg_dr: None
        :type dg_dr: None

        :return: the predicted properties
        :rtype: numpy array of shape (n_samples,)
        """

        approved_x, approved_classes = self._check_predict_input(x, classes, dgdr)

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            model = graph.get_tensor_by_name("Model/output:0")
            y_pred = self.session.run(model, feed_dict = {tf_x : approved_x})
            return y_pred

    # TODO these need to be checked if they still work
    def save_nn(self, save_dir="./saved_model"):
        """
        This function saves the model to be used for later prediction.

        :param save_dir: name of the directory to create to save the model
        :type save_dir: string
        :return: None
        """
        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        if not os.path.exists(save_dir):
            pass
        else:
            ii = 1
            while True:
                new_save_dir = save_dir + "_" + str(ii)
                if not os.path.exists(new_save_dir):
                    save_dir = new_save_dir
                    break

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
    The ``ARMP`` class is used to build neural networks that take as an input atomic representations of molecules and
    output molecular properties such as the energies.
    """

    def __init__(self, hidden_layer_sizes = (5,), l1_reg = 0.0, l2_reg = 0.0001, batch_size = 'auto', learning_rate = 0.001,
                 iterations = 500, tensorboard = False, store_frequency = 200, tf_dtype = tf.float32, scoring_function = 'mae',
                 activation_function = "sigmoid", optimiser = tf.train.AdamOptimizer, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08,
                 rho = 0.95, initial_accumulator_value = 0.1, initial_gradient_squared_accumulator_value = 0.1,
                 l1_regularization_strength = 0.0, l2_regularization_strength = 0.0,
                 tensorboard_subdir = os.getcwd() + '/tensorboard', representation_name='acsf', representation_params=None):
        """
        To see what parameters are required, look at the description of the _NN class init.
        This class inherits from the _NN class and all inputs not unique to the ARMP class are passed to the _NN
        parent.
        """

        super(ARMP, self).__init__(hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                 iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                 activation_function, optimiser, beta1, beta2, epsilon,
                 rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                 l1_regularization_strength,l2_regularization_strength, tensorboard_subdir)

        self._initialise_representation(representation_name, representation_params)

    def _initialise_representation(self, representation, parameters):
        """
        This function sets the representation and the parameters of the representation.

        :param representation: the name of the representation
        :type representation: string
        :param parameters: all the parameters of the representation.
        :type parameters: dictionary
        :return: None
        """

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['slatm', 'acsf']:
            raise InputError("Unknown representation %s" % representation)
        self.representation_name = representation.lower()

        if parameters is not None:
            if not type(parameters) is dict:
                raise InputError("The representation parameters passed should be either None or a dictionary.")
            self._check_representation_parameters(parameters)

        if self.representation_name == 'slatm':

            self._set_slatm_parameters(parameters)

        elif self.representation_name == 'acsf':

            self._set_acsf_parameters(parameters)

        else:

            if not is_none(parameters):
                raise InputError("The representation %s does not take any additional parameters." % (self.representation_name))

    def _set_representation(self, g):

        if len(g.shape) != 3:
            raise InputError(
                "The representation should have a shape (n_samples, n_atoms, n_features). Got %s" % (str(g.shape)))

        self.g = g

    def _generate_representations_from_data(self, xyz, classes, method):
        """
        This function generates the representations from xyz data

        :param xyz: the cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: classes to use for atomic decomposition
        :type classes: numpy array of shape (n_samples, n_atoms)
        :param method: What method to use
        :type method: string
        :return: representations and classes
        :rtype: numpy arrays of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """

        if method not in ['fortran', 'tf']:
            raise InputError("The method to generate the acsf can only be 'fortran' or 'tf'. Got %s." % (method))

        if isinstance(classes, type(None)):
            raise InputError("The classes need to be provided for the ARMP estimator.")
        else:
            if len(classes.shape) > 2 or np.all(xyz.shape[:2] != classes.shape):
                raise InputError("Classes should be a 2D array with shape matching the first 2 dimensions of the xyz data"
                                 ". Got shape %s" % (str(classes.shape)))

        representation = None

        if self.representation_name == 'slatm':
            # TODO implement
            raise InputError("Slatm from data has not been implemented yet. Use Compounds.")

        elif self.representation_name == 'acsf':
            if method == 'tf':
                representation = self._generate_acsf_from_data_tf(xyz, classes)
            else:
                representation = self._generate_acsf_from_data_fortran(xyz, classes)

        return representation, classes

    def _generate_acsf_from_data_tf(self, xyz, classes):
        """
        This function generates the acsf from the cartesian coordinates and the classes.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: the classes to use for atomic decomposition
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: representation acsf
        :rtype: numpy array of shape (n_samples, n_atoms, n_features)
        """

        if 0 in classes:
            idx_zeros = np.where(classes == 0)[1]
            classes_for_elements = classes[:, :idx_zeros[0]]
        else:
            classes_for_elements = classes

        elements, element_pairs = self._get_elements_and_pairs(classes_for_elements)

        if self.tensorboard:
            self.tensorboard_logger_representation.initialise()

        n_samples = xyz.shape[0]
        max_n_atoms = xyz.shape[1]

        # Turning the quantities into tensors
        with tf.name_scope("Inputs"):
            zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
            xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
            dataset = dataset.batch(20)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_xyz, batch_zs = iterator.get_next()

        representation = generate_parkhill_acsf(xyzs=batch_xyz, Zs=batch_zs, elements=elements, element_pairs=element_pairs,
                                            rcut=self.acsf_parameters['rcut'],
                                            acut=self.acsf_parameters['acut'],
                                            nRs2=self.acsf_parameters['nRs2'],
                                            nRs3=self.acsf_parameters['nRs3'],
                                            nTs=self.acsf_parameters['nTs'],
                                            eta=self.acsf_parameters['eta'],
                                            zeta=self.acsf_parameters['zeta'])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyz, zs_tf: classes})

        representation_slices = []

        if self.tensorboard:
            self.tensorboard_logger_representation.set_summary_writer(sess)

            batch_counter = 0
            while True:
                try:
                    representation_np = sess.run(representation, options=self.tensorboard_logger_representation.options,
                                             run_metadata=self.tensorboard_logger_representation.run_metadata)
                    self.tensorboard_logger_representation.write_metadata(batch_counter)
                    representation_slices.append(representation_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    print("Generated all the representations.")
                    break
        else:
            while True:
                try:
                    representation_np = sess.run(representation)
                    representation_slices.append(representation_np)
                except tf.errors.OutOfRangeError:
                    break

        representation_conc = np.concatenate(representation_slices, axis=0)

        sess.close()

        return representation_conc

    def _generate_acsf_from_data_fortran(self, xyz, classes):
        """
        This function uses fortran to generate the representation and the derivative of the representation with respect
        to the cartesian coordinates.
        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: the atom types
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: representations and their derivatives wrt to xyz
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms, n_features, n_atoms, 3)
        """

        initial_natoms = xyz.shape[1]

        elements, _ = self._get_elements_and_pairs(classes)

        representation = []

        for i in range(xyz.shape[0]):
            if 0 in classes[i]:
                idx_zeros = np.where(classes == 0)[1]
                mol_xyz = xyz[i, :idx_zeros[0], :]
                mol_classes = classes[i, :idx_zeros[0]]

                g = generate_acsf(coordinates=mol_xyz, elements=elements, gradients=False, nuclear_charges=mol_classes,
                                  rcut=self.acsf_parameters['rcut'],
                                  acut=self.acsf_parameters['acut'],
                                  nRs2=self.acsf_parameters['nRs2'],
                                  nRs3=self.acsf_parameters['nRs3'],
                                  nTs=self.acsf_parameters['nTs'],
                                  eta2=self.acsf_parameters['eta'],
                                  eta3=self.acsf_parameters['eta'],
                                  zeta=self.acsf_parameters['zeta'])

                padded_g = np.zeros((initial_natoms, g.shape[-1]))
                padded_g[:g.shape[0], :] = g

                representation.append(padded_g)

            else:

                g = generate_acsf(coordinates=xyz[i], elements=elements, gradients=False, nuclear_charges=classes[i],
                                  rcut=self.acsf_parameters['rcut'],
                                  acut=self.acsf_parameters['acut'],
                                  nRs2=self.acsf_parameters['nRs2'],
                                  nRs3=self.acsf_parameters['nRs3'],
                                  nTs=self.acsf_parameters['nTs'],
                                  eta2=self.acsf_parameters['eta'],
                                  eta3=self.acsf_parameters['eta'],
                                  zeta=self.acsf_parameters['zeta'])

                representation.append(g)

        return np.asarray(representation)

    def _generate_representations_from_compounds(self):
        """
        This function generates the representations from the compounds.
        :return: the representations and the classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """

        if self.compounds is None:
            raise InputError("QML compounds needs to be created in advance")

        if self.representation_name == 'slatm':

            representations, classes = self._generate_slatm_from_compounds()

        elif self.representation_name == 'acsf':

            representations, classes = self._generate_acsf_from_compounds()

        else:
            raise InputError("This should never happen, unrecognised representation %s." % (self.representation_name))

        return representations, classes

    def _generate_acsf_from_compounds(self):
        """
        This function generates the atom centred symmetry functions.

        :return: representation acsf and classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """

        # Obtaining the xyz and the nuclear charges
        xyzs = []
        zs = []
        max_n_atoms = 0

        for compound in self.compounds:
            xyzs.append(compound.coordinates)
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        elements, element_pairs = self._get_elements_and_pairs(zs)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            zs_padding = np.zeros(missing_n_atoms)
            zs[i] = np.concatenate((zs[i], zs_padding))
            xyz_padding = np.zeros((missing_n_atoms, 3))
            xyzs[i] = np.concatenate((xyzs[i], xyz_padding))

        zs = np.asarray(zs, dtype=np.int32)
        xyzs = np.asarray(xyzs, dtype=np.float32)

        if self.tensorboard:
            self.tensorboard_logger_representation.initialise()
            # run_metadata = tf.RunMetadata()
            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        # Turning the quantities into tensors
        with tf.name_scope("Inputs"):
            zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
            xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
            dataset = dataset.batch(20)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_xyz, batch_zs = iterator.get_next()

        representations = generate_parkhill_acsf(xyzs=batch_xyz, Zs=batch_zs, elements=elements, element_pairs=element_pairs,
                                                 rcut=self.acsf_parameters['rcut'],
                                                 acut=self.acsf_parameters['acut'],
                                                 nRs2=self.acsf_parameters['nRs2'],
                                                 nRs3=self.acsf_parameters['nRs3'],
                                                 nTs=self.acsf_parameters['nTs'],
                                                 eta=self.acsf_parameters['eta'],
                                                 zeta=self.acsf_parameters['zeta'])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyzs, zs_tf: zs})

        representations_slices = []

        if self.tensorboard:
            self.tensorboard_logger_representation.set_summary_writer(sess)

            batch_counter = 0
            while True:
                try:
                    representations_np = sess.run(representations, options=self.tensorboard_logger_representation.options,
                                             run_metadata=self.tensorboard_logger_representation.run_metadata)
                    self.tensorboard_logger_representation.write_metadata(batch_counter)

                    representations_slices.append(representations_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
        else:
            batch_counter = 0
            while True:
                try:
                    representations_np = sess.run(representations)
                    representations_slices.append(representations_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break

        representation_conc = np.concatenate(representations_slices, axis=0)

        sess.close()

        return representation_conc, zs

    def _generate_slatm_from_compounds(self):
        """
        This function generates the slatm using the data in the compounds.

        :return: representation slatm and the classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features) and (n_samples, n_atoms)
        """
        mbtypes = qml_rep.get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
        list_representations = []
        max_n_atoms = 0

        # Generating the representation in the shape that ARMP requires it
        for compound in self.compounds:
            compound.generate_slatm(mbtypes, local=True, sigmas=[self.representation_params['slatm_sigma1'],
                                                                 self.representation_params['slatm_sigma2']],
                                    dgrids=[self.representation_params['slatm_dgrid1'],
                                            self.representation_params['slatm_dgrid2']],
                                    rcut=self.representation_params['slatm_rcut'],
                                    alchemy=self.representation_params['slatm_alchemy'],
                                    rpower=self.representation_params['slatm_rpower'])
            representation = compound.representation
            if max_n_atoms < representation.shape[0]:
                max_n_atoms = representation.shape[0]
            list_representations.append(representation)

        # Padding the representations of the molecules that have fewer atoms
        n_samples = len(list_representations)
        n_features = list_representations[0].shape[1]
        padded_representations = np.zeros((n_samples, max_n_atoms, n_features))
        for i, item in enumerate(list_representations):
            padded_representations[i, :item.shape[0], :] = item

        # Generating zs in the shape that ARMP requires it
        zs = np.zeros((n_samples, max_n_atoms))
        for i, mol in enumerate(self.compounds):
            zs[i, :mol.nuclear_charges.shape[0]] = mol.nuclear_charges

        return padded_representations, zs

    def _atomic_model(self, x, hidden_layer_sizes, weights, biases):
        """
        Constructs the atomic part of the network. It calculates the output for all atoms as if they all were the same
        element.

        :param x: Atomic representation
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

        :param x: Atomic representation
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


        all_atomic_energies = tf.zeros_like(zs, dtype=tf.float32)

        for el in self.elements:
            # Obtaining the indices of where in Zs there is the current element
            current_element = tf.expand_dims(tf.constant(el, dtype=tf.int32), axis=0)
            where_element = tf.cast(tf.where(tf.equal(zs, current_element)), dtype=tf.int32)

            # Extract the descriptor corresponding to the right element
            current_element_in_x = tf.gather_nd(x, where_element)

            # Calculate the atomic energy of all the atoms of type equal to the current element
            atomic_ene = self._atomic_model(current_element_in_x, self.hidden_layer_sizes, element_weights[el],
                                                 element_biases[el])

            # Put the atomic energies in a zero array with shape equal to zs and then add it to all the atomic energies
            updates = tf.scatter_nd(where_element, atomic_ene, tf.shape(zs))
            all_atomic_energies = tf.add(all_atomic_energies, updates)

        # Summing the energies of all the atoms
        total_energies = tf.reduce_sum(all_atomic_energies, axis=-1, name="output", keepdims=True)

        return total_energies

    def _cost(self, y_pred, y, weights_dict):
        """
        This function calculates the cost function during the training of the neural network.

        :param y_pred: the neural network predictions
        :type y_pred: tensors of shape (n_samples, 1)
        :param y: the truth values
        :type y: tensors of shape (n_samples, 1)
        :param weights_dict: the dictionary containing all of the weights
        :type weights_dict: dictionary where the key is a nuclear charge and the value is a list of tensors of length hidden_layer_sizes.size + 1
        :return: value of the cost function
        :rtype: tf.Variable of size (1,)
        """

        err =  tf.square(tf.subtract(y, y_pred))
        cost_function = tf.reduce_mean(err, name="loss")

        if self.l2_reg >= 0:
            l2_loss = 0
            for element in weights_dict:
                l2_loss += self._l2_loss(weights_dict[element])
            cost_function += l2_loss
        if self.l1_reg >= 0:
            l1_loss = 0
            for element in weights_dict:
                l1_loss += self._l1_loss(weights_dict[element])
            cost_function += l1_loss

        return cost_function

    def _check_inputs(self, x, y, classes, dy, dgdr):
        """
        This function checks that all the needed input data is available.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param classes: classes to use for the atomic decomposition
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dy: None
        :type dy: None
        :param dgdr: None
        :type dgdr: None

        :return: the approved x, y dy and classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features), (n_samples, 1), None, (n_samples, n_atoms)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if dy is not None:
            raise InputError("ARMP estimator cannot be used to predict gradients. Use ARMP_G estimator.")

        if not is_none(dgdr):
            raise InputError("ARMP estimator cannot be used to predict gradients. Use ARMP_G estimator.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.g):

                if self.compounds is None:
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.g, self.classes = self._generate_representations_from_compounds()

            if self.properties is None:
                raise InputError("The properties need to be set in advance.")
            if self.classes is None:
                raise InputError("The classes need to be set in advance.")

            approved_x = self.g[x]
            approved_y = self._get_properties(x)
            approved_dy = None
            approved_classes = self.classes[x]

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        else:

            if y is None:
                raise InputError("y cannot be of None type.")
            if classes is None:
                raise InputError("ARMP estimator needs the classes to do atomic decomposition.")

            approved_x = check_local_representation(x)
            approved_y = check_y(y)
            approved_dy = None
            approved_classes = check_classes(classes)

            check_sizes(approved_x, approved_y, approved_dy, approved_classes)

        return approved_x, approved_y, approved_dy, approved_classes

    def _check_predict_input(self, x, classes, dgdr):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_atoms, n_features)
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: None
        :type dg_dr: None

        :return: the approved representation and classes
        :rtype: numpy array of shape (n_samples, n_atoms, n_features), (n_samples, n_atoms)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        if not is_none(dgdr):
            raise InputError("ARMP does not require gradients of the representation.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.g):
                if is_none(self.compounds):
                    raise InputError("No representations or QML compounds have been set yet.")
                else:
                    self.g, self.classes = self._generate_representations_from_compounds()
            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")

            approved_x = self.g[x]
            approved_classes = self.classes[x]

            check_sizes(x=approved_x, classes=approved_classes)

        else:

            if isinstance(classes, type(None)):
                raise InputError("ARMP estimator needs the classes to do atomic decomposition.")

            approved_x = check_local_representation(x)
            approved_classes = check_classes(classes)

            check_sizes(x=approved_x, classes=approved_classes)

        return approved_x, approved_classes

    def _check_representation_parameters(self, parameters):
        """
        This function checks that the dictionary passed that contains parameters of the representation contains the right
        parameters.

        :param parameters: all the parameters of the representation.
        :type parameters: dictionary
        :return: None
        """

        if self.representation_name == "slatm":

            slatm_parameters = {'slatm_sigma1': 0.05, 'slatm_sigma2': 0.05, 'slatm_dgrid1': 0.03, 'slatm_dgrid2': 0.03,
                                'slatm_rcut': 4.8, 'slatm_rpower': 6, 'slatm_alchemy': False}

            for key, value in parameters.items():
                try:
                    slatm_parameters[key]
                except Exception:
                    raise InputError("Unrecognised parameter for slatm representation: %s" % (key))

        elif self.representation_name == "acsf":

            acsf_parameters =  {'rcut': 5.0, 'acut': 5.0, 'nRs2': 5, 'nRs3': 5, 'nTs': 5,
                                      'zeta': 220.127, 'eta': 30.8065}

            for key, value in parameters.items():
                try:
                    acsf_parameters[key]
                except Exception:
                    raise InputError("Unrecognised parameter for acsf representation: %s" % (key))

    def _get_elements_and_pairs(self, classes):
        """
        This function generates the atom centred symmetry functions.
        :param classes: The different types of atoms present in the system
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: elements and element pairs in the system
        :rtype: numpy array of shape (n_elements,) and (n_element_pairs)
        """

        elements = np.unique(classes)
        elements_no_zero = np.ma.masked_equal(elements,0).compressed()

        element_pairs = []
        for i, ei in enumerate(elements_no_zero):
            for ej in elements_no_zero[i:]:
                element_pairs.append([ej, ei])

        return np.asarray(elements_no_zero), np.asarray(element_pairs)

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

    def _fit(self, x, y, classes, dy, dgdr):
        """
        This function calls either fit_from_scratch or fit_from_loaded depending on if a model has been loaded or not.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None

        :return: None
        """

        if not self.loaded_model:
            self._fit_from_scratch(x, y, dy, classes)
        else:
            self._fit_from_loaded(x, y, dy, classes)

    def _fit_from_scratch(self, x, y, dy, classes):
        """
        This function fits an atomic decomposed network to the data.

        :param x: either the representations or the indices to the data points to use
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dy: None
        :type dy: None
        :param dg_dr: None
        :type dg_dr: None

        :return: None
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, None)

        # Putting a mask on all the 0 values
        classes_for_elements = np.ma.masked_equal(classes_approved, 0).compressed()

        self.elements, self.element_pairs = self._get_elements_and_pairs(classes_for_elements)

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()

        # Useful quantities
        self.n_samples = x_approved.shape[0]
        self.n_atoms = x_approved.shape[1]
        self.n_features = x_approved.shape[2]

        # Set the batch size
        batch_size = self._get_batch_size()

        # This is the total number of batches in which the training set is divided
        n_batches = ceil(self.n_samples, batch_size)

        tf.reset_default_graph()

        # Initial set up of the NN
        with tf.name_scope("Data"):
            x_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, self.n_atoms, self.n_features], name="Descriptors")
            zs_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.n_atoms], name="Atomic-numbers")
            y_ph = tf.placeholder(dtype=self.tf_dtype, shape=[None, 1], name="Properties")
            buffer_tf = tf.placeholder(dtype=tf.int64, name="buffer")

            dataset = tf.data.Dataset.from_tensor_slices((x_ph, zs_ph, y_ph))
            dataset = dataset.shuffle(buffer_size=buffer_tf)
            dataset = dataset.batch(batch_size)
            # batched_dataset = dataset.prefetch(buffer_size=batch_size)

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            tf_x, tf_zs, tf_y = iterator.get_next()

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
                    self.tensorboard_logger_training.write_weight_histogram(weights)

        with tf.name_scope("Model"):
            molecular_energies = self._model(tf_x, tf_zs, element_weights, element_biases)

        with tf.name_scope("Cost_func"):
            cost = self._cost(molecular_energies, tf_y, element_weights)

        if self.tensorboard:
            cost_summary = self.tensorboard_logger_training.write_cost_summary(cost)

        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost, name="optimisation_op")

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        iterator_init = iterator.make_initializer(dataset, name="dataset_init")

        self._build_model_from_xyz(self.n_atoms, element_weights, element_biases)

        self.session = tf.Session()

        # Running the graph
        if self.tensorboard:
            self.tensorboard_logger_training.set_summary_writer(self.session)

        self.session.run(init)

        # Initialising the object that enables graceful killing of the training
        killer = GracefulKiller()

        for i in range(self.iterations):

            if i % 2 == 0:
                buff = int(3.5 * batch_size)
            else:
                buff = int(4.5 * batch_size)

            self.session.run(iterator_init, feed_dict={x_ph: x_approved, zs_ph: classes_approved, y_ph: y_approved, buffer_tf:buff})
            avg_cost = 0

            for j in range(n_batches):
                if self.tensorboard:
                    opt, c = self.session.run([optimisation_op, cost], options=self.tensorboard_logger_training.options,
                             run_metadata=self.tensorboard_logger_training.run_metadata)
                else:
                    opt, c = self.session.run([optimisation_op, cost])

                avg_cost += c

                if killer.kill_now:
                    self.save_nn("emergency_save")
                    exit()

            # This seems to run the iterator.get_next() op, which gives problems with end of sequence, hence why I re-initialise the iterator
            if self.tensorboard:
                if i % self.tensorboard_logger_training.store_frequency == 0:
                    self.session.run(iterator_init,
                                     feed_dict={x_ph: x_approved, zs_ph: classes_approved, y_ph: y_approved,
                                                buffer_tf: buff})
                    self.tensorboard_logger_training.write_summary(self.session, i)

            self.training_cost.append(avg_cost/n_batches)

    def _fit_from_loaded(self, x, y, dy, classes):
        """
       This function carries on fitting an atomic decomposed network to the data after it has been loaded.

       :param x: either the representations or the indices to the data points to use
       :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
       :param y: either the properties or None
       :type y: either a numpy array of shape (n_samples,) or None
       :param dy: None
       :type dy: None
       :param classes: classes to use for the atomic decomposition or None
       :type classes: either a numpy array of shape (n_samples, n_atoms) or None

       :return: None
       """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, dy, classes, dgdr=None)

        # Putting a mask on all the 0 values
        classes_for_elements = np.ma.masked_equal(classes_approved, 0).compressed()

        self.elements, self.element_pairs = self._get_elements_and_pairs(classes_for_elements)

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()
            self.tensorboard_logger_training.set_summary_writer(self.session)

        self.n_samples = x_approved.shape[0]
        self.n_atoms = x_approved.shape[1]
        self.n_features = x_approved.shape[2]

        batch_size = self._get_batch_size()
        n_batches = ceil(self.n_samples, batch_size)

        graph = tf.get_default_graph()

        with graph.as_default():
            # Reloading all the needed operations and tensors
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            tf_ene = graph.get_tensor_by_name("Data/Properties:0")
            tf_buffer = graph.get_tensor_by_name("Data/buffer:0")

            optimisation_op = graph.get_operation_by_name("optimisation_op")
            dataset_init_op = graph.get_operation_by_name("dataset_init")

        # Initialising the object that enables graceful killing of the training
        killer = GracefulKiller()

        for i in range(self.iterations):

            if i % 2 == 0:
                buff = int(3.5 * batch_size)
            else:
                buff = int(4.5 * batch_size)

            self.session.run(dataset_init_op, feed_dict={tf_x: x_approved, tf_zs: classes_approved, tf_ene: y_approved, tf_buffer: buff})

            for j in range(n_batches):
                if self.tensorboard:
                    self.session.run(optimisation_op, options=self.tensorboard_logger_training.options,
                                     run_metadata=self.tensorboard_logger_training.run_metadata)
                else:
                    self.session.run(optimisation_op)

                if killer.kill_now:
                    self.save_nn("emergency_save")
                    exit()

            if self.tensorboard:
                if i % self.tensorboard_logger_training.store_frequency == 0:
                    self.session.run(dataset_init_op,
                                     feed_dict={tf_x: x_approved, tf_zs: classes_approved, tf_ene: y_approved, tf_buffer: buff})
                    self.tensorboard_logger_training.write_summary(self.session, i)

    def _build_model_from_xyz(self, n_atoms, element_weights, element_biases):
        """
        This function builds a model that makes it possible to predict energies straight from xyz data. It constructs the
        graph needed to do this.

        :param n_atoms: number of atoms
        :param element_weights: the dictionary of the trained weights in the model
        :param element_biases: the dictionary of trained biases in the model
        """

        with tf.name_scope("Inputs_pred"):
            zs_tf = tf.placeholder(shape=[None, n_atoms], dtype=tf.int32, name="Classes")
            xyz_tf = tf.placeholder(shape=[None, n_atoms, 3], dtype=tf.float32, name="xyz")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
            dataset = dataset.batch(2)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_xyz, batch_zs = iterator.get_next()
            iterator_init = iterator.make_initializer(dataset, name="dataset_init_pred")

        with tf.name_scope("Descriptor_pred"):
            batch_representation = generate_parkhill_acsf(xyzs=batch_xyz, Zs=batch_zs, elements=self.elements,
                                                          element_pairs=self.element_pairs,
                                                          rcut=self.acsf_parameters['rcut'],
                                                          acut=self.acsf_parameters['acut'],
                                                          nRs2=self.acsf_parameters['nRs2'],
                                                          nRs3=self.acsf_parameters['nRs3'],
                                                          nTs=self.acsf_parameters['nTs'],
                                                          eta=self.acsf_parameters['eta'],
                                                          zeta=self.acsf_parameters['zeta'])

        with tf.name_scope("Model_pred"):
            batch_energies_nn = self._model(batch_representation, batch_zs, element_weights, element_biases)

    def _predict(self, x, classes, dgdr):
        """
        This function checks whether x contains indices or data. If it contains indices, the data is extracted by the
        appropriate compound objects. Otherwise it checks what data is passed through the arguments. Then, the data is
        used as input to the trained network to predict some properties.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or of floats of shape (n_samples, n_atoms, n_features)
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: None
        :type dg_dr: None

        :return: the predicted properties
        :rtype: numpy array of shape (n_samples,)
        """

        approved_x, approved_classes = self._check_predict_input(x, classes, dgdr)
        empty_ene = np.empty((approved_x.shape[0], 1))

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            tf_true_ene = graph.get_tensor_by_name("Data/Properties:0")
            tf_buffer =  graph.get_tensor_by_name("Data/buffer:0")
            model = graph.get_tensor_by_name("Model/output:0")
            dataset_init_op = graph.get_operation_by_name("dataset_init")
            self.session.run(dataset_init_op, feed_dict={tf_x: approved_x, tf_zs: approved_classes, tf_true_ene: empty_ene, tf_buffer:1})

        tot_y_pred = []

        while True:
            try:
                y_pred = self.session.run(model)
                tot_y_pred.append(y_pred)
            except tf.errors.OutOfRangeError:
                break

        return np.concatenate(tot_y_pred, axis=0)

    def predict_from_xyz(self, xyz, classes):
        """
        This function takes in the cartesian coordinates and the atom types and returns energies.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: atom types
        :type classes: numpy array of shape (n_samples, n_atoms)
        :return: energies
        :rtype: numpy array of shape  (n_samples,)
        """

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            xyz_tf = graph.get_tensor_by_name("Inputs_pred/xyz:0")
            classes_tf = graph.get_tensor_by_name("Inputs_pred/Classes:0")
            ene_nn = graph.get_tensor_by_name("Model_pred/output:0")
            dataset_init_op = graph.get_operation_by_name("Inputs_pred/dataset_init_pred")
            self.session.run(dataset_init_op,
                             feed_dict={xyz_tf: xyz, classes_tf: classes})

        tot_y_pred = []

        while True:
            try:
                y_pred = self.session.run(ene_nn)
                tot_y_pred.append(y_pred)
            except tf.errors.OutOfRangeError:
                break

        return np.concatenate(tot_y_pred, axis=0).ravel()

    def _score_r2(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: None
        :type dg_dr: None

        :return: R^2
        :rtype: float
        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        y_pred = self.predict(x_approved, classes_approved)
        r2 = r2_score(y_approved, y_pred, sample_weight = None)
        return r2

    def _score_mae(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: None
        :type dg_dr: None

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Mean absolute error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        y_pred = self.predict(x_approved, classes_approved)
        mae = (-1.0) * mean_absolute_error(y_approved, y_pred, sample_weight=None)
        print("Warning! The mae is multiplied by -1 so that it can be minimised in Osprey!")
        return mae

    def _score_rmse(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: None
        :type dy: None
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: None
        :type dg_dr: None

        :return: Root mean square error
        :rtype: float

        """

        x_approved, y_approved, dy_approved, classes_approved = self._check_inputs(x, y, classes, dy, dgdr)

        y_pred = self.predict(x_approved, classes_approved)
        rmse = np.sqrt(mean_squared_error(y_approved, y_pred, sample_weight = None))
        return rmse

    def save_nn(self, save_dir="saved_model"):
        """
        This function saves the trained model to be used for later prediction.

        :param save_dir: directory in which to save the model
        :type save_dir: string
        :return: None
        """

        counter = 0
        dir = save_dir
        while True:
            if os.path.isdir(save_dir):
                counter += 1
                save_dir = dir + "_" + str(counter)
            else:
                break

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            tf_x = graph.get_tensor_by_name("Data/Descriptors:0")
            tf_zs = graph.get_tensor_by_name("Data/Atomic-numbers:0")
            true_ene = graph.get_tensor_by_name("Data/Properties:0")
            model = graph.get_tensor_by_name("Model/output:0")

        tf.saved_model.simple_save(self.session, export_dir=save_dir,
                                   inputs={"Data/Descriptors:0": tf_x, "Data/Atomic-numbers:0": tf_zs,
                                           "Data/Properties:0": true_ene},
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

        self.loaded_model=True

### --------------------- ** Atomic representation - molecular properties with gradients ** ----------------------------

class ARMP_G(ARMP, _NN):
    """
    The ``ARMP_G`` class  is used to build neural networks that take as an input atomic representations of molecules and
    output molecular properties and their gradients, like the energies and the forces.
    """

    def __init__(self, hidden_layer_sizes=(5,), l1_reg=0.0, l2_reg=0.0001, batch_size='auto', learning_rate=0.001,
                 iterations=500, tensorboard=False, store_frequency=200, tf_dtype=tf.float32, scoring_function='mae',
                 activation_function="sigmoid", optimiser=tf.train.AdamOptimizer, beta1=0.9, beta2=0.999,
                 epsilon=1e-08,
                 rho=0.95, initial_accumulator_value=0.1, initial_gradient_squared_accumulator_value=0.1,
                 l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                 tensorboard_subdir=os.getcwd() + '/tensorboard', representation_name='acsf', representation_params=None,
                 phi=1.0, forces_score_weight=0.0):

        super(ARMP_G, self).__init__(hidden_layer_sizes, l1_reg, l2_reg, batch_size, learning_rate,
                                   iterations, tensorboard, store_frequency, tf_dtype, scoring_function,
                                   activation_function, optimiser, beta1, beta2, epsilon,
                                   rho, initial_accumulator_value, initial_gradient_squared_accumulator_value,
                                   l1_regularization_strength, l2_regularization_strength, tensorboard_subdir)

        if representation_name != 'acsf':
            raise InputError("Only the acsf representation can currently be used with gradients.")

        self._initialise_representation(representation_name, representation_params)
        self._set_phi(phi)
        self._set_forces_score_weight(forces_score_weight)

    def _set_phi(self, phi):
        """
        Sets the parameter that weights the forces term in the cost function. If phi = 0, then it will be equivalent to
        doing the calculation with ARMP, except much slower since the gradients are still calculated.

        :param phi: parameter that multiplies the forces in the cost function
        :type phi: positive float
        :return: None
        """

        if is_positive_or_zero(phi):
            self.phi = phi
        else:
            raise InputError("Phi should be positive or zero.")

    def _set_forces_score_weight(self, forces_score_weight):
        """
        Sets the parameter that weights the forces term in the score function.
        If forces_score_weight = 0, then it will be equivalent to only score the energy predictions.
        If forces_score_weight = 1, then it will be equivalent to only score the forces predictions.

        :param forces_score_weight: Weight of the forces in the scoring
        :type forces_score_weight: float
        :return: None
        """

        if is_positive_or_zero(forces_score_weight) and forces_score_weight <= 1:
            self.forces_score_weight = forces_score_weight
        else:
            raise InputError("forces_score_weight should be between zero and one")

    def _check_inputs(self, x, y, classes, dy, dgdr):
        """
        This function checks that the data passed to the fit function makes sense. If X represent indices, it extracts
        the data from the variables self.xyz, self.properties, self.gradients and self.classes. If the representations have
        been generated prior to calling this function, it uses self.g and self.dg_dr instead of self.xyz.
        Otherwise the representations and their gradients wrt the Cartesian coordinates are generated.

        :param x: Indices or the cartesian coordinates
        :type x: Either 1D numpy array of ints or numpy array of floats of shape (n_samples, n_atoms, 3)
        :param y: The properties - for example the molecular energies (or None if x represents indices)
        :type y: numpy array of shape (n_samples,)
        :param classes: The different types of atoms in the system (or None if x represents indices)
        :type classes: numpy array of shape (n_samples, n_atoms)
        :param dy: Gradients of the molecular properties - for example the forces (or None if x represents indices)
        :type dy: numpy array of shape (n_samples, n_atoms, 3)
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: The representation, the properties, classes, gradients, the gradients of the representation wrt xyz
        :rtype: (n_samples, n_atoms, n_features), (n_samples,), (n_samples, n_atoms), (n_samples, n_atoms, 3),
        (n_samples, n_atoms, n_features, n_atoms, 3)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.xyz) or is_none(self.classes):
                if not is_none(self.compounds):
                    self.xyz = self._get_xyz_from_compounds(x)
                    self.classes = self._get_classes_from_compounds(x)
                    approved_xyz = self.xyz[x]
                    approved_classes = self.classes[x]
                else:
                    raise InputError("The xyz coordinates and the classes need to have been set in advance.")
            else:
                approved_xyz = self.xyz[x]
                approved_classes = self.classes[x]

            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")
            else:
                approved_y = self._get_properties(x)

            if is_none(self.gradients):
                raise InputError("The gradients need to be set in advance.")
            else:
                approved_dy = self.gradients[x]

        else:
            if is_none(y):
                raise InputError("y cannot be of None type.")
            if is_none(dy):
                raise InputError("ARMP_G estimator requires gradients.")
            if is_none(classes):
                raise InputError("ARMP_G estimator needs the classes to do atomic decomposition.")

            approved_xyz = check_xyz(x)
            approved_y = check_y(y)
            approved_dy = check_dy(dy)
            approved_classes = check_classes(classes)

        check_sizes(approved_xyz, approved_y, approved_dy, approved_classes)

        return approved_xyz, approved_y, approved_classes, approved_dy

    def _check_predict_input(self, x, classes, dgdr):
        """
        This function has the same role as _check_inputs except it does not check y and dy.

        :param x: indices or data
        :type x: numpy array of ints of shape (n_samples,) or floats of shape (n_samples, n_atoms, n_features)
        :param classes: classes to use for the atomic decomposition or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates or None
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: The representation, the gradients of the representation wrt xyz and classes
        :rtype: (n_samples, n_atoms, n_features), (n_samples, n_atoms, n_features, n_atoms, 3), (n_samples, n_atoms)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.xyz) or is_none(self.classes):
                if not is_none(self.compounds):
                    idx_tot = len(self.compounds)
                    self.xyz = self._get_xyz_from_compounds(idx_tot)
                    self.classes = self._get_classes_from_compounds(idx_tot)
                    approved_xyz = self.xyz[x]
                    approved_classes = self.classes[x]
                else:
                    raise InputError("The xyz coordinates and the classes need to have been set in advance.")
            else:
                approved_xyz = self.xyz[x]
                approved_classes = self.classes[x]
            check_sizes(x=approved_xyz, classes=approved_classes)

        else:

            if is_none(classes):
                raise InputError("ARMP_G needs the classes to do atomic decomposition for predictions.")

            approved_xyz = check_xyz(x)
            approved_classes = check_classes(classes)

            check_sizes(x=approved_xyz, classes=approved_classes)

        return approved_xyz, approved_classes

    def _check_score_input(self, x, y, dy):
        """
        This function checks that the data passed to the fit function makes sense. If X represent indices, it extracts
        the data from the variables self.properties, self.gradients.

        :param x: Indices or the cartesian coordinates
        :type x: Either 1D numpy array of ints or numpy array of floats of shape (n_samples, n_atoms, 3)
        :param y: The properties - for example the molecular energies (or None if x represents indices)
        :type y: numpy array of shape (n_samples,)
        :param dy: Gradients of the molecular properties - for example the forces (or None if x represents indices)
        :type dy: numpy array of shape (n_samples, n_atoms, 3)
        :return: properties, gradients
        :rtype: (n_samples,), (n_samples, n_atoms, 3)
        """

        if not is_array_like(x):
            raise InputError("x should be an array either containing indices or data.")

        # Check if x is made up of indices or data
        if is_positive_integer_or_zero_array(x):

            if is_none(self.properties):
                raise InputError("The properties need to be set in advance.")
            else:
                approved_y = self._get_properties(x)

            if is_none(self.gradients):
                raise InputError("The gradients need to be set in advance.")
            else:
                approved_dy = self.gradients[x]

        else:
            if is_none(y):
                raise InputError("y cannot be of None type.")
            if is_none(dy):
                raise InputError("ARMP_G estimator requires gradients.")

            approved_y = check_y(y)
            approved_dy = check_dy(dy)

        return approved_y, approved_dy

    def _make_weights_biases(self, train_elements):
        """
        This function uses the self.elements data to initialise tensors of weights and biases for each element present
        in the system.

        :return: dictionaries of weights and biases for each element
        :rtype: two dictionaries where the keys are ints and the value are tensors
        """
        element_weights = {}
        element_biases = {}

        with tf.name_scope("Weights"):
            for i in range(train_elements.shape[0]):
                weights, biases = self._generate_weights(n_out=1)
                element_weights[train_elements[i]] = weights
                element_biases[train_elements[i]] = biases

                # Log weights for tensorboard
                if self.tensorboard:
                    self.tensorboard_logger_training.write_weight_histogram(weights)

        return element_weights, element_biases

    def _cost_G(self, y_true, y_nn, dy_true, dy_nn, weights_dict):
        """
        This function calculates the cost for the ARMP_G class. It uses both true energies/forces and the neural network
        predicted energies/forces.

        :param y_true: True properties
        :type y_true: tf tensor of shape (n_sample,)
        :param y_nn: Neural network predicted properties
        :type y_nn: tf tensor of shape (n_sample,)
        :param dy_true: True gradients
        :type dy_true: tf tensor of shape (n_sample, n_atoms, 3)
        :param dy_nn: Neural network predicted gradients
        :type dy_nn: tf tensor of shape (n_sample, n_atoms, 3)
        :param weights_dict: dictionary containing the weights for each element specific network.
        :return: tf.tensor of shape ()
        """

        ene_err = tf.square(tf.subtract(y_true, y_nn))
        force_err = tf.square(tf.subtract(dy_true, dy_nn))
        phi_tf = tf.constant(self.phi, dtype=tf.float32)

        cost_function = tf.add(tf.reduce_mean(ene_err), tf.reduce_mean(force_err)*phi_tf, name="loss")

        if self.l2_reg >= 0:
            l2_loss = 0
            for element in weights_dict:
                l2_loss += self._l2_loss(weights_dict[element])
            cost_function += l2_loss
        if self.l1_reg >= 0:
            l1_loss = 0
            for element in weights_dict:
                l1_loss += self._l1_loss(weights_dict[element])
            cost_function += l1_loss

        return cost_function

    def _fit(self, x, y, classes, dy, dgdr):
        """
        This fit function checks whether there is a model that has already been loaded. If yes, it calls the fit function
        that restarts training from where it was left off. Otherwise, the fitting is started from scratch.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param y: molecular properties
        :type y: numpy array of shape (n_samples,)
        :param dy: gradients of the properties wrt to cartesian coordinates
        :type dy: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: type of the atoms in the system
        :type classes: numpy array of shape (n_samples, n_atoms)
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)
        :return: None
        """
        if not self.loaded_model:
            self._fit_from_scratch(x, y, classes, dy)
        else:
            self._fit_from_loaded(x, y, classes, dy)

    def _fit_from_loaded(self, x, y, classes, dy):

        if self.session == None:
            raise InputError("The Tensorflow session appears to not exisit.")

        xyz_approved, y_approved, classes_approved, dy_approved = self._check_inputs(x, y, classes, dy, None)

        if is_none(self.element_pairs) and is_none(self.elements):

            classes_for_elements = np.ma.masked_equal(classes_approved, 0).compressed()
            self.elements, self.element_pairs = self._get_elements_and_pairs(classes_for_elements)

        self.n_samples = xyz_approved.shape[0]
        max_n_atoms = xyz_approved.shape[1]

        batch_size = self._get_batch_size()
        n_batches = ceil(self.n_samples, batch_size)

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()
            self.tensorboard_logger_training.set_summary_writer(self.session)

        graph = tf.get_default_graph()

        # Initialising the object that enables graceful killing of the training
        killer = GracefulKiller()

        with graph.as_default():
            # Reloading all the needed operations and tensors
            xyz_tf = graph.get_tensor_by_name("Data/xyz:0")
            zs_tf = graph.get_tensor_by_name("Data/Classes:0")
            true_ene = graph.get_tensor_by_name("Data/Properties:0")
            true_forces = graph.get_tensor_by_name("Data/Forces:0")
            tf_buffer = graph.get_tensor_by_name("Data/buffer:0")

            optimisation_op = graph.get_operation_by_name("optimisation_op")
            dataset_init_op = graph.get_operation_by_name("dataset_init")

            # Running the operations needed
            for i in range(self.iterations):

                if i % 2 == 0:
                    buff = int(3.5 * batch_size)
                else:
                    buff = int(4.5 * batch_size)

                self.session.run(dataset_init_op,
                                 feed_dict={xyz_tf: xyz_approved, zs_tf: classes_approved, true_ene: y_approved,
                                            true_forces: dy_approved, tf_buffer: buff})

                for j in range(n_batches):
                    if self.tensorboard:
                        self.session.run(optimisation_op, options=self.tensorboard_logger_training.options,
                                         run_metadata=self.tensorboard_logger_training.run_metadata)
                    else:
                        self.session.run(optimisation_op)

                    if killer.kill_now:
                        self.save_nn("emergency_save")
                        exit()

                if self.tensorboard:
                    if i % self.tensorboard_logger_training.store_frequency == 0:
                        self.session.run(dataset_init_op,
                                         feed_dict={xyz_tf: xyz_approved, zs_tf: classes_approved, true_ene: y_approved,
                                                    true_forces: dy_approved, tf_buffer: buff})
                        self.tensorboard_logger_training.write_summary(self.session, i)

    def _fit_from_scratch(self, x, y, classes, dy):
        """
        This function fits the weights of the neural networks to the properties and their gradient from scratch.

        :param xyz: cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param y: molecular properties
        :type y: numpy array of shape (n_samples,)
        :param dy: gradients of the properties wrt to cartesian coordinates
        :type dy: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: type of the atoms in the system
        :type classes: numpy array of shape (n_samples, n_atoms)
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)
        :return: None
        """

        xyz_approved, y_approved, classes_approved, dy_approved = self._check_inputs(x, y, classes, dy, None)

        if is_none(self.element_pairs) and is_none(self.elements):
            classes_for_elements = np.ma.masked_equal(classes_approved, 0).compressed()
            self.elements, self.element_pairs = self._get_elements_and_pairs(classes_for_elements)

            self.n_features = self.elements.shape[0] * self.acsf_parameters['nRs2'] + \
                              self.element_pairs.shape[0] * self.acsf_parameters['nRs3'] * \
                              self.acsf_parameters['nTs']

        self.n_samples = xyz_approved.shape[0]
        max_n_atoms = xyz_approved.shape[1]

        batch_size = self._get_batch_size()
        n_batches = ceil(self.n_samples, batch_size)

        if self.tensorboard:
            self.tensorboard_logger_training.initialise()

        element_weights, element_biases = self._make_weights_biases(self.elements)

        # Turning the quantities into tensors
        with tf.name_scope("Data"):
            zs_tf = tf.placeholder(shape=[None, max_n_atoms], dtype=tf.int32, name="Classes")
            xyz_tf = tf.placeholder(shape=[None, max_n_atoms, 3], dtype=tf.float32, name="xyz")
            true_ene = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Properties")
            true_forces = tf.placeholder(shape=[None, max_n_atoms, 3], dtype=tf.float32, name="Forces")
            buffer_tf = tf.placeholder(dtype=tf.int64, name="buffer")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf, true_ene, true_forces))
            dataset = dataset.shuffle(buffer_size=buffer_tf)
            dataset = dataset.batch(batch_size).prefetch(2)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_xyz, batch_zs, batch_y, batch_dy = iterator.get_next()

        with tf.name_scope("Descriptor"):
            batch_g = generate_parkhill_acsf(batch_xyz, batch_zs, self.elements, self.element_pairs,
                                               rcut=self.acsf_parameters['rcut'],
                                               acut=self.acsf_parameters['acut'],
                                               nRs2=self.acsf_parameters['nRs2'],
                                               nRs3=self.acsf_parameters['nRs3'],
                                               nTs=self.acsf_parameters['nTs'],
                                               eta=self.acsf_parameters['eta'],
                                               zeta=self.acsf_parameters['zeta'])

        # Creating the model
        with tf.name_scope("Model"):
            energies_nn = self._model(batch_g, batch_zs, element_weights, element_biases)
            forces_nn = - tf.gradients(energies_nn, batch_xyz)[0]

        # Calculating the cost
        with tf.name_scope("Cost"):
            cost = self._cost_G(batch_y, energies_nn, batch_dy, forces_nn, element_weights)

        if self.tensorboard:
            cost_summary = self.tensorboard_logger_training.write_cost_summary(cost)

        optimiser = self._set_optimiser()
        optimisation_op = optimiser.minimize(cost, name="optimisation_op")

        # Initialisation of variables and iterators
        init = tf.global_variables_initializer()
        iterator_init = iterator.make_initializer(dataset, name="dataset_init")

        # Starting the session
        self.session = tf.Session()

        if self.tensorboard:
            self.tensorboard_logger_training.set_summary_writer(self.session)

        self.session.run(init)

        for i in range(self.iterations):

            if i % 2 == 0:
                buff = int(3.5 * batch_size)
            else:
                buff = int(4.5 * batch_size)

            self.session.run(iterator_init,
                             feed_dict={xyz_tf: xyz_approved, zs_tf: classes_approved, true_ene: y_approved,
                                        true_forces: dy_approved, buffer_tf:buff})

            for j in range(n_batches):
                if self.tensorboard:
                    self.session.run(optimisation_op, options=self.tensorboard_logger_training.options,
                             run_metadata=self.tensorboard_logger_training.run_metadata)
                else:
                    self.session.run(optimisation_op)

            # This seems to run the iterator.get_next() op, which gives problems with end of sequence
            # Hence why I re-initialise the iterator
            if self.tensorboard:
                if i % self.tensorboard_logger_training.store_frequency == 0:
                    self.session.run(iterator_init,
                                     feed_dict={xyz_tf: xyz_approved, zs_tf: classes_approved, true_ene: y_approved,
                                                true_forces: dy_approved, buffer_tf:buff})
                    self.tensorboard_logger_training.write_summary(self.session, i)

    def predict(self, x, classes=None, dgdr=None):
        """
        This function overwrites the parent predict, because it needs to return not only the properties but also the
        gradients.

        :param x: representation or indices
        :type x: numpy array of shape (n_samples, n_features) or (n_samples, n_atoms, n_features) or an array of ints
        :param classes: the classes to use for atomic decomposition
        :type classes: numpy array of shape (n_sample, n_atoms)
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: predictions of the molecular properties and their gradients.
        :rtype: numpy array of shape (n_samples,) and (n_samples, n_atoms, 3)
        """
        prop_predictions, grad_predictions = self._predict(x, classes, dgdr)

        if prop_predictions.ndim > 1 and prop_predictions.shape[1] == 1:
            prop_predictions = prop_predictions.ravel()

        return prop_predictions, grad_predictions

    def _predict(self, x, classes, dgdr):
        """
        This function predicts the properties and their gradient starting from the cartesian coordinates and the atom
        types.

        :param xyz: Cartesian coordinates
        :type xyz: numpy array of shape (n_samples, n_atoms, 3)
        :param classes: the different atom types present in the system
        :type classes: numpy array of shape (n_samples, n_atoms)
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: predicted properties and their gradients
        :rtype: numpy arrays of shape (n_samples,) and (n_samples, n_atoms, 3)
        """

        xyz_approved, classes_approved = self._check_predict_input(x, classes, dgdr)

        # TODO find a cleaner way of doing this (surgery?)
        empty_ene = np.empty((xyz_approved.shape[0], 1))
        empty_forces = np.empty((xyz_approved.shape[0], xyz_approved.shape[1], 3))

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            xyz_tf = graph.get_tensor_by_name("Data/xyz:0")
            zs_tf = graph.get_tensor_by_name("Data/Classes:0")
            model = graph.get_tensor_by_name("Model/output:0")
            output_grad = graph.get_tensor_by_name("Model/Neg:0")
            true_ene = graph.get_tensor_by_name("Data/Properties:0")
            true_forces = graph.get_tensor_by_name("Data/Forces:0")
            tf_buffer = graph.get_tensor_by_name("Data/buffer:0")

            dataset_init_op = graph.get_operation_by_name("dataset_init")

            self.session.run(dataset_init_op, feed_dict={xyz_tf: xyz_approved, zs_tf:classes_approved,
                                                         true_ene: empty_ene, true_forces: empty_forces, tf_buffer:1})

            tot_y_pred = []
            tot_dy_pred = []

            while True:
                try:
                    y_pred, dy_pred = self.session.run([model, output_grad])
                    tot_y_pred.append(y_pred)
                    tot_dy_pred.append(dy_pred)
                except tf.errors.OutOfRangeError:
                    break

        return np.concatenate(tot_y_pred, axis=0), np.concatenate(tot_dy_pred, axis=0)

    def _score_r2(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the coefficient of determination (R^2).
        Larger values corresponds to a better prediction.

        :param x: either the cartesian coordinates or the indices to the samples
        :type x: either a numpy array of shape (n_samples, n_atoms, 3) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients or None
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3)
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: average R^2 of the properties and the gradient
        :rtype: float
        """

        y_approved, dy_approved = self._check_score_input(x, y, dy)

        y_pred, dy_pred = self.predict(x, classes)

        y_r2 = r2_score(y_approved, y_pred, sample_weight = None)
        dy_approved = np.reshape(dy_approved, (dy_approved.shape[0], dy_approved.shape[1]*dy_approved.shape[2]))
        dy_pred = np.reshape(dy_pred, (dy_pred.shape[0], dy_pred.shape[1] * dy_pred.shape[2]))
        dy_r2 = r2_score(dy_approved, dy_pred, sample_weight= None)
        r2 = (1 - self.forces_score_weight) * y_r2 + dy_r2 * self.forces_score_weight
        return r2

    def _score_mae(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the mean absolute error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients or None
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3)
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :param sample_weight: Weights of the samples. None indicates that that each sample has the same weight.
        :type sample_weight: array of shape (n_samples,)

        :return: Average Mean absolute error of the properties and the gradient
        :rtype: float
        """

        y_approved, dy_approved = self._check_score_input(x, y, dy)

        y_pred, dy_pred = self.predict(x, classes)

        dy_approved = np.reshape(dy_approved, (dy_approved.shape[0], dy_approved.shape[1] * dy_approved.shape[2]))
        dy_pred = np.reshape(dy_pred, (dy_pred.shape[0], dy_pred.shape[1] * dy_pred.shape[2]))
        y_mae = mean_absolute_error(y_approved, y_pred, sample_weight=None)
        dy_mae = mean_absolute_error(dy_approved, dy_pred, sample_weight=None)
        mae = (1 - self.forces_score_weight) * y_mae + dy_mae * self.forces_score_weight
        return mae

    def _score_rmse(self, x, y=None, classes=None, dy=None, dgdr=None):
        """
        Calculate the root mean squared error.
        Smaller values corresponds to a better prediction.

        :param x: either the representations or the indices to the representations
        :type x: either a numpy array of shape (n_samples, n_atoms, n_features) or a numpy array of ints
        :param y: either the properties or None
        :type y: either a numpy array of shape (n_samples,) or None
        :param dy: either the gradients or None
        :type dy: either a numpy array of shape (n_samples, n_atoms, 3)
        :param classes: either the classes or None
        :type classes: either a numpy array of shape (n_samples, n_atoms) or None
        :param dg_dr: gradients of the representation with respect to the cartesian coordinates
        :type dg_dr: numpy array of shape (n_samples, n_atoms, n_features, n_atoms, 3)

        :return: Average root mean square error of the properties and the gradient
        :rtype: float
        """

        y_approved, dy_approved = self._check_score_input(x, y, dy)

        y_pred, dy_pred = self.predict(x, classes)
        dy_approved = np.reshape(dy_approved, (dy_approved.shape[0], dy_approved.shape[1] * dy_approved.shape[2]))
        dy_pred = np.reshape(dy_pred, (dy_pred.shape[0], dy_pred.shape[1] * dy_pred.shape[2]))
        y_rmse = np.sqrt(mean_squared_error(y_approved, y_pred, sample_weight = None))
        dy_rmse = np.sqrt(mean_squared_error(dy_approved, dy_pred, sample_weight=None))
        rmse = (1 - self.forces_score_weight) * y_rmse + dy_rmse * self.forces_score_weight
        return rmse

    # TODO modify so that it inherits from ARMP
    def save_nn(self, save_dir="saved_model"):
        """
        This function saves the trained model to be used for later prediction.

        :param save_dir: directory in which to save the model
        :type save_dir: string
        :return: None
        """

        counter = 0
        dir = save_dir
        while True:
            if os.path.isdir(save_dir):
                counter += 1
                save_dir = dir + "_" + str(counter)
            else:
                break

        if self.session == None:
            raise InputError("Model needs to be fit before predictions can be made.")

        graph = tf.get_default_graph()

        with graph.as_default():
            xyz = graph.get_tensor_by_name("Data/xyz:0")
            zs = graph.get_tensor_by_name("Data/Classes:0")
            true_ene = graph.get_tensor_by_name("Data/Properties:0")
            true_forces = graph.get_tensor_by_name("Data/Forces:0")
            model = graph.get_tensor_by_name("Model/output:0")
            model_grad = graph.get_tensor_by_name("Model/Neg:0")

        tf.saved_model.simple_save(self.session, export_dir=save_dir,
                                   inputs={"Data/xyz:0": xyz, "Data/Classes:0":zs, "Data/Properties:0":true_ene,
                                           "Data/Forces:0":true_forces},
                                   outputs={"Model/output:0": model, "Model/Neg:0":model_grad})

    def load_nn(self, save_dir="saved_model"):
        """
        This function reloads a model for predictions.

        :param save_dir: the name of the directory where the model is saved.
        :type save_dir: string
        :return: None
        """

        self.session = tf.Session(graph=tf.get_default_graph())
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], save_dir)

        self.loaded_model = True

