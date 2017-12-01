"""
Main module where all the neural network magic happens
"""
# Temporary fix until pip package


from __future__ import print_function
#from __future__ import absolute_import
import os
import sys
#sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
#from sklearn.utils.validation import check_X_y, check_array
import tensorflow as tf
#import inverse_dist as inv
#import matplotlib.pyplot as plt
#from sklearn.metrics import r2_score
#from tensorflow.python.framework import ops
#from tensorflow.python.training import saver as saver_lib
#from tensorflow.python.framework import graph_io
#from tensorflow.python.tools import freeze_graph

#TODO relative imports
from utils import is_positive, is_positive_integer, \
        is_positive_integer_or_zero, is_bool, is_string, is_positive_or_zero

class _NN(BaseEstimator, RegressorMixin):

    """
    Parent class for training multi-layered neural networks on molecular or atomic properties via Tensorflow
    """

    def __init__(self, hidden_layer_sizes = [5], l1_reg = 0.0, l2_reg = 0.0001, batch_size = 'auto', learning_rate = 0.001,
                 iterations = 80, tensorboard = False, store_frequency = 200, 
                 tensorboard_subdir = os.getcwd() + '/tensorboard', **args):
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
        self.hidden_layer_sizes = hidden_layer_sizes
        self._process_hidden_layer_sizes()

        if not is_positive(l1_reg):
            raise ValueError("Expected positive float value for variable l1_reg. Got %s" % str(l1_reg))
        self.l1_reg = l1_reg

        if not is_positive(l2_reg):
            raise ValueError("Expected positive float value for variable l2_reg. Got %s" % str(l2_reg))
        self.l2_reg = l2_reg

        if batch_size != "auto":
            if not is_positive_integer(batch_size):
                raise ValueError("Expected batch_size to be a positive integer. Got %s" % str(batch_size))
            self.batch_size = int(batch_size)
        else:
            self.batch_size = batch_size

        if not is_positive(learning_rate):
            raise ValueError("Expected positive float value for variable learning_rate. Got %s" % str(learning_rate))
        self.learning_rate = float(learning_rate)

        if not is_positive_integer(iterations):
            raise ValueError("Expected positive integer value for variable iterations. Got %s" % str(iterations))
        self.iterations = float(iterations)

        if not is_bool(tensorboard):
            raise ValueError("Expected boolean value for variable tensorboard. Got %s" % str(tensorboard))
        self.tensorboard = bool(tensorboard)

        if tensorboard:
            if not is_positive_integer(store_frequency):
                raise ValueError("Expected positive integer value for variable store_frequency. Got %s" % str(store_frequency))
            if store_frequency > self.iterations:
                print("Only storing final iteration for tensorboard")
                self.store_frequency = self.iterations

        self.tensorboard_subdir = tensorboard_subdir
        # Creating tensorboard directory if tensorflow flag is on
        if self.tensorboard:
            if not is_string(self.tensorboard_subdir):
                raise ValueError('Expected string value for variable tensorboard_subdir. Got %s' % str(self.tensorboard_subdir))
            if not os.path.exists(self.tensorboard_subdir):
                os.makedirs(self.tensorboard_subdir)


        # Placeholder variables
        self.n_features = None
        self.n_samples = None
        self.train_cost = []
        self.test_cost = []
        self.model_exist = False
        #self.loaded_model = False
        #self.is_vis_ready = False

    def _process_hidden_layers(self):
        hidden_layer_sizes = []
        for i,n in enumerate(self.hidden_layer_sizes):
            if not is_positive_integer_or_zero(n):
                raise ValueError("Hidden layer size must be a positive integer. Got %s" % str(n))

            # Ignore layers of size zero
            if int(n) == 0:
                break
            hidden_layer_sizes.append(int(n))

        if len(hidden_layer_sizes) == 0:
            raise ValueError("Hidden layers must be non-zero. Got %s" % str(hidden_layer_sizes))
        
        self.hidden_layer_sizes = hidden_layer_sizes

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
        weights.append(tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[0], self.n_features], stddev=1.0/np.sqrt(n.features)),
                                   name='weight_in'))
        biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[0]]), name='bias_in'))

        # Weights from one hidden layer to the next
        for ii in range(len(self.hidden_layer_sizes) - 1):
            weights.append(tf.Variable(
                tf.truncated_normal([self.hidden_layer_sizes[ii + 1], self.hidden_layer_sizes[ii]], stddev=1.0/np.sqrt(self.hidden_layer_sizes[ii])),
                name='weight_hidden_%d' % ii))
            biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[ii + 1]]), name='bias_hidden_%d' % ii))

        # Weights from last hidden layer to output layer
        weights.append(tf.Variable(tf.truncated_normal([n_out, self.hidden_layer_sizes[-1]], stddev=0.1), name='weight_out'))
        biases.append(tf.Variable(tf.zeros([n_out]), name='bias_out'))

        return weights, biases

    def _l2_reg(self, weights):
        """
        Creates the expression for L2-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l2_regu")

        reg_term += tf.reduce_sum(tf.square(weights))

        return reg_term

    def _l1_reg(self, weights):
        """
        Creates the expression for L1-regularisation on the weights

        :param weights: tensorflow tensors representing the weights
        :type weights: list
        :return: tensorflow scalar representing the regularisation contribution to the cost function
        :rtype: tf.float32
        """

        reg_term = tf.zeros([], name="l1_weight")

        reg_term += tf.reduce_sum(tf.abs(weights))

        return reg_term

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
            raise TypeError("Wrong data type of variable 'filename'. Expected string")

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
            raise ValueError("Shape mismatch between predicted and true values. %s and %s" % (str(y_nn.shape), str(y_true.shape)))

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
                raise TypeError("Wrong data type of variable 'filename'. Expected string")
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
                    raise TypeError("Wrong data type of variable 'filename'. Expected string")


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
        X, y = check_X_y(X, self.properties, multi_output=False)

        # Flag that a model has been trained
        self.model_exist = True

        # Useful quantities
        self.n_coord = X.shape[1]
        self.n_samples = X.shape[0]
        self.n_atoms = int(X.shape[1]/3)

        # Check the value of the batch size
        self.batch_size = self.checkBatchSize()

        # Place holders for the input/output data
        with tf.name_scope('Data'):
            in_data = tf.placeholder(tf.float32, [None, self.n_coord], name="Coordinates")
            out_data = tf.placeholder(tf.float32, [None, self.n_coord + 1], name="Energy_forces")

        # Making the descriptor from the Cartesian coordinates
        with tf.name_scope('Descriptor'):
            X_des = self.available_descriptors[self.descriptor](in_data, n_atoms=self.n_atoms)

        # Number of features in the descriptor
        self.n_features = int(self.n_atoms * (self.n_atoms - 1) * 0.5)

        # Randomly initialisation of the weights and biases
        with tf.name_scope('weights'):
            weights, biases = self.__generate_weights(n_out=(1+3*self.n_atoms))

            # Log weights for tensorboard
            if self.tensorboard:
                tf.summary.histogram("weights_in", weights[0])
                for ii in range(len(self.hidden_layer_sizes) - 1):
                    tf.summary.histogram("weights_hidden", weights[ii + 1])
                tf.summary.histogram("weights_out", weights[-1])


        # Calculating the output of the neural net
        with tf.name_scope('model'):
            out_NN = self.modelNN(X_des, weights, biases)

        # Obtaining the derivative of the neural net energy wrt cartesian coordinates
        with tf.name_scope('grad_ene'):
            ene_NN = tf.slice(out_NN,begin=[0,0], size=[-1,1], name='ene_NN')
            grad_ene_NN = tf.gradients(ene_NN, in_data, name='dEne_dr')[0] * (-1)

        # Calculating the cost function
        with tf.name_scope('cost_funct'):
            err_ene_force = tf.square(tf.subtract(out_NN, out_data), name='err2_ene_force')
            err_grad = tf.square(tf.subtract(tf.slice(out_data, begin=[0,1], size=[-1,-1]), grad_ene_NN), name='err2_grad')

            cost_ene_force = tf.reduce_mean(err_ene_force, name='cost_ene_force')
            cost_grad = tf.reduce_mean(err_grad, name='cost_grad')

            reg_term = self.__reg_term(weights)

            cost = cost_ene_force + self.alpha_grad*cost_grad + self.alpha_reg * reg_term

        # Training the network
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        if self.tensorboard:
            merged_summary = tf.summary.merge_all()
            options = tf.RunOptions()
            options.output_partition_graphs = True
            options.trace_level = tf.RunOptions.SOFTWARE_TRACE
            run_metadata = tf.RunMetadata()

        # Running the graph
        with tf.Session() as sess:
            if self.tensorboard:
                summary_writer = tf.summary.FileWriter(logdir=self.board_dir,graph=sess.graph)
            sess.run(init)

            for iter in range(self.max_iter):
                # This is the total number of batches in which the training set is divided
                n_batches = int(self.n_samples / self.batch_size)
                # This will be used to calculate the average cost per iteration
                avg_cost = 0
                # Learning over the batches of data
                for i in range(n_batches):
                    batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
                    batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
                    opt, c = sess.run([optimizer, cost], feed_dict={in_data: batch_x, out_data: batch_y})
                    avg_cost += c / n_batches

                    if self.tensorboard:
                        if iter % self.print_step == 0:
                            # The options flag is needed to obtain profiling information
                            summary = sess.run(merged_summary, feed_dict={in_data: batch_x, out_data: batch_y}, options=options, run_metadata=run_metadata)
                            summary_writer.add_summary(summary, iter)
                            summary_writer.add_run_metadata(run_metadata, 'iteration %d batch %d' % (iter, i))

                self.trainCost.append(avg_cost)

            # Saving the weights for later re-use
            self.all_weights = []
            self.all_biases = []
            for ii in range(len(weights)):
                self.all_weights.append(sess.run(weights[ii]))
                self.all_biases.append(sess.run(biases[ii]))



#    an energy term, a force term *and* a gradient term in the cost function.
#    
#    The input to the fit function is the matrix of n_samples by n_coordinates (no atom labels). This is then transformed
#    into a descriptor which is fed into the neural network. There is only 1 neural network which has an output of size
#    n_atoms x3 + 1 (3 n_atoms forces, plus the energy).
#    """
#
#        :alpha_grad: default 0.05
#            Parameter that enables to tweak the importance of the gradient term in the cost function.


#    def save(self, path):
#        """
#        Stores a .meta, .index, .data_0000-0001 and a check point file, which can be used to restore
#        the trained model.
#
#        :param path: Path of directory where files are stored. The path is assumed to be absolute
#                     unless there are no forward slashes (/) in the path name.
#        :type path: string
#        """
#
#        if self.is_trained == False:
#            raise Exception("The fit function has not been called yet, so the model can't be saved.")
#
#        # Creating a new graph
#        model_graph = tf.Graph()
#
#        with model_graph.as_default():
#            # Making the placeholder for the data
#            xyz_test = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")
#
#            # Making the descriptor from the Cartesian coordinates
#            X_des = self.available_descriptors[self.descriptor](xyz_test, n_atoms=self.n_atoms)
#
#            # Setting up the trained weights
#            weights = []
#            biases = []
#
#            for ii in range(len(self.all_weights)):
#                weights.append(tf.Variable(self.all_weights[ii], name="weights_restore"))
#                biases.append(tf.Variable(self.all_biases[ii], name="biases_restore"))
#
#            # Calculating the ouputs
#            out_NN = self.modelNN(X_des, weights, biases)
#
#            init = tf.global_variables_initializer()
#
#            # Object needed to save the model
#            all_saver = tf.train.Saver(save_relative_paths=True)
#
#            with tf.Session() as sess:
#                sess.run(init)
#
#                # Saving the graph
#                all_saver.save(sess, dir)
#    def load_NN(self, dir):
#        """
#        Function that loads a trained estimator.
#
#        :dir: directory where the .meta, .index, .data_0000-0001 and check point files have been saved.
#        """
#
#        # Inserting the weights into the model
#        with tf.Session() as sess:
#            # Loading a saved graph
#            file = dir + ".meta"
#            saver = tf.train.import_meta_graph(file)
#
#            # The model is loaded in the default graph
#            graph = tf.get_default_graph()
#
#            # Loading the graph of out_NN
#            self.out_NN = graph.get_tensor_by_name("output_node:0")
#            self.in_data = graph.get_tensor_by_name("Cartesian_coord:0")
#
#            saver.restore(sess, dir)
#            sess.run(tf.global_variables_initializer())
#
#        self.loadedModel = True

#    def predict(self, X):
#        """
#        This function uses the X data and plugs it into the model and then returns the predicted y.
#
#        :X: array of shape (n_samples, n_features)
#            This contains the input data with samples in the rows and features in the columns.
#
#        :return: array of size (n_samples, n_outputs)
#            This contains the predictions for the target values corresponding to the samples contained in X.
#        """
#
#        check_array(X)
#
#        if self.alreadyInitialised:
#
#            # MAking the placeholder for the data
#            xyz_test = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")
#
#            # Making the descriptor from the Cartesian coordinates
#            X_des = self.available_descriptors[self.descriptor](xyz_test, n_atoms=self.n_atoms)
#
#            # Setting up the trained weights
#            weights = []
#            biases = []
#
#            for ii in range(len(self.all_weights)):
#                weights.append(tf.Variable(self.all_weights[ii]))
#                biases.append(tf.Variable(self.all_biases[ii]))
#
#            # Calculating the ouputs
#            out_NN = self.modelNN(X_des, weights, biases)
#
#            init = tf.global_variables_initializer()
#
#            with tf.Session() as sess:
#                sess.run(init)
#                ene_forces_NN = sess.run(out_NN, feed_dict={xyz_test: X})
#
#            return ene_forces_NN
#
#        elif self.loadedModel:
#            feed_dic = {self.in_data: X}
#
#            with tf.Session() as sess:
#                sess.run(tf.global_variables_initializer())
#                ene_forces_NN = sess.run(self.out_NN, feed_dict=feed_dic)
#
#            return ene_forces_NN
#
#        else:
#            raise Exception("The fit function has not been called yet, so the model has not been trained yet.")

# def fit_and_predict

#    def modelNN(self, X, weights, biases):
#        """
#        Specifies how the output is calculated from the input.
#
#        :X: tf.placeholder of shape (n_samples, n_features)
#        :weights: list of tf.Variables of length len(hidden_layer_sizes) + 1
#        :biases: list of tf.Variables of length len(hidden_layer_sizes) + 1
#        :return: tf.Variable of size (n_samples, n_outputs)
#        """
#
#        # Calculating the activation of the first hidden layer
#        z = tf.add(tf.matmul(X, tf.transpose(weights[0])), biases[0])
#        h = tf.nn.sigmoid(z)
#
#        # Calculating the activation of all the hidden layers
#        for ii in range(len(self.hidden_layer_sizes)-1):
#            z = tf.add(tf.matmul(h, tf.transpose(weights[ii+1])), biases[ii+1])
#            h = tf.nn.sigmoid(z)
#
#        # Calculating the output of the last layer
#        z = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1], name="output_node")
#
#        return z

#    def score(self, X, y, sample_weight=None):
#        """
#        Returns the pearson correlation coefficient . It calculates the R^2 value. It is used during the
#        training of the model.
#
#        :X: array of shape (n_samples, n_features)
#
#            This contains the input data with samples in the rows and features in the columns.
#
#        :y: array of shape (n_samples, n_outputs)
#
#            This contains the target values for each sample in the X matrix.
#
#        :sample_weight: array of shape (n_samples,)
#
#            Sample weights (not sure what this is, but i need it for inheritance from the BaseEstimator)
#
#        :return: double
#            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.
#        """
#
#        y_pred = self.predict(X)
#        r2 = r2_score(y, y_pred)
#        return r2

#    def fit(self, X, y):
#        """
#        Fit the  to data matrix X and target y.
#
#        :X: array of shape (n_samples, n_features).
#
#            This contains the input data with samples in the rows and features in the columns.
#
#        :y: array of shape (n_samples, n_outputs).
#
#            This contains the target values for each sample in the X matrix.
#
#        """
#
#        # Check that X and y have correct shape
#        X, y = check_X_y(X, y, multi_output=True)
#
#        self.alreadyInitialised = True
#
#        # Useful quantities
#        self.n_coord = X.shape[1]
#        self.n_samples = X.shape[0]
#        self.n_atoms = int(X.shape[1]/3)
#
#        # Check the value of the batch size
#        self.batch_size = self.checkBatchSize()
#
#        # Place holders for the input/output data
#        with tf.name_scope('Data'):
#            in_data = tf.placeholder(tf.float32, [None, self.n_coord], name="Coordinates")
#            out_data = tf.placeholder(tf.float32, [None, self.n_coord + 1], name="Energy_forces")
#
#        # Making the descriptor from the Cartesian coordinates
#        with tf.name_scope('Descriptor'):
#            X_des = self.available_descriptors[self.descriptor](in_data, n_atoms=self.n_atoms)
#
#        # Number of features in the descriptor
#        self.n_feat = int(self.n_atoms * (self.n_atoms - 1) * 0.5)
#
#        # Randomly initialisation of the weights and biases
#        with tf.name_scope('weights'):
#            weights, biases = self.__generate_weights(n_out=(1+3*self.n_atoms))
#            
#            # Log weights for tensorboard
#            if self.tensorboard:
#                tf.summary.histogram("weights_in", weights[0])
#                for ii in range(len(self.hidden_layer_sizes) - 1):
#                    tf.summary.histogram("weights_hidden", weights[ii + 1])
#                tf.summary.histogram("weights_out", weights[-1])
#
#
#        # Calculating the output of the neural net
#        with tf.name_scope('model'):
#            out_NN = self.modelNN(X_des, weights, biases)
#
#        # Obtaining the derivative of the neural net energy wrt cartesian coordinates
#        with tf.name_scope('grad_ene'):
#            ene_NN = tf.slice(out_NN,begin=[0,0], size=[-1,1], name='ene_NN')
#            grad_ene_NN = tf.gradients(ene_NN, in_data, name='dEne_dr')[0] * (-1)
#
#        # Calculating the cost function
#        with tf.name_scope('cost_funct'):
#            err_ene_force = tf.square(tf.subtract(out_NN, out_data), name='err2_ene_force')
#            err_grad = tf.square(tf.subtract(tf.slice(out_data, begin=[0,1], size=[-1,-1]), grad_ene_NN), name='err2_grad')
#
#            cost_ene_force = tf.reduce_mean(err_ene_force, name='cost_ene_force')
#            cost_grad = tf.reduce_mean(err_grad, name='cost_grad')
#
#            reg_term = self.__reg_term(weights)
#
#            cost = cost_ene_force + self.alpha_grad*cost_grad + self.alpha_reg * reg_term
#
#        # Training the network
#        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)
#
#        if self.tensorboard:
#            cost_summary = tf.summary.scalar('cost', cost)
#
#        # Initialisation of the variables
#        init = tf.global_variables_initializer()
#        if self.tensorboard:
#            merged_summary = tf.summary.merge_all()
#            options = tf.RunOptions()
#            options.output_partition_graphs = True
#            options.trace_level = tf.RunOptions.SOFTWARE_TRACE
#            run_metadata = tf.RunMetadata()
#
#        # Running the graph
#        with tf.Session() as sess:
#            if self.tensorboard:
#                summary_writer = tf.summary.FileWriter(logdir=self.board_dir,graph=sess.graph)
#            sess.run(init)
#
#            for iter in range(self.max_iter):
#                # This is the total number of batches in which the training set is divided
#                n_batches = int(self.n_samples / self.batch_size)
#                # This will be used to calculate the average cost per iteration
#                avg_cost = 0
#                # Learning over the batches of data
#                for i in range(n_batches):
#                    batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
#                    batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
#                    opt, c = sess.run([optimizer, cost], feed_dict={in_data: batch_x, out_data: batch_y})
#                    avg_cost += c / n_batches
#
#                    if self.tensorboard:
#                        if iter % self.print_step == 0:
#                            # The options flag is needed to obtain profiling information
#                            summary = sess.run(merged_summary, feed_dict={in_data: batch_x, out_data: batch_y}, options=options, run_metadata=run_metadata)
#                            summary_writer.add_summary(summary, iter)
#                            summary_writer.add_run_metadata(run_metadata, 'iteration %d batch %d' % (iter, i))
#
#                self.trainCost.append(avg_cost)
#
#            # Saving the weights for later re-use
#            self.all_weights = []
#            self.all_biases = []
#            for ii in range(len(weights)):
#                self.all_weights.append(sess.run(weights[ii]))
#                self.all_biases.append(sess.run(biases[ii]))

#class _FFNN(_MLPRegFlow):
#    """
#    Parent class for feed forward neural networks where either a molecular representation is
#    used to predict a molecular property (e.g. energies) or a local representation is used
#    to predict an atomic property (e.g. chemical shieldings).
#    """
#
#class _AFFNN(_MLPRegFlow):
#    """
#    Parent class for feed forward neural networks where atomic 
#    used to predict a molecular property (e.g. energies) or a local representation is used
#    to predict an atomic property (e.g. chemical shieldings).
#    """




if __name__ == "__main__":
    lol = MRMP()
