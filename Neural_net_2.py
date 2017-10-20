"""
This file contains a class of a neural network that fits both the energies and the forces at the same time. It uses
an energy term, a force term *and* a gradient term in the cost function.

The input to the fit function is the matrix of n_samples by n_coordinates (no atom labels). This is then transformed
into a descriptor which is fed into the neural network. There is only 1 neural network which has an output of size
n_atoms x3 + 1 (3 n_atoms forces, plus the energy).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import tensorflow as tf
import inverse_dist as inv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib
import os
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

class MLPRegFlow(BaseEstimator, ClassifierMixin):


    def __init__(self, hidden_layer_sizes=(5,), alpha_reg=0.0001, alpha_grad=0.05, batch_size='auto', learning_rate_init=0.001,
                 max_iter=80, hl1=0, hl2=0, hl3=0, descriptor="inverse_dist"):
        """
        Neural-network with multiple hidden layers to do regression.

        :hidden_layer_sizes: Tuple, length = number of hidden layers, default (5,).
            The ith element represents the number of neurons in the ith
            hidden layer.
        :alpha_reg: float, default 0.0001
            L2 penalty (regularization term) parameter.
        :alpha_grad: default 0.05
            Parameter that enables to tweak the importance of the gradient term in the cost function.
        :batch_size: int, default 'auto'.
            Size of minibatches for stochastic optimizers.
            If the solver is 'lbfgs', the classifier will not use minibatch.
            When set to "auto", `batch_size=min(200, n_samples)`
        :learning_rate_init: double, default 0.001.
            The value of the learning rate in the numerical minimisation.
        :max_iter: int, default 200.
            Total number of iterations that will be carried out during the training process.
        :hl1: int, default 0
            Number of neurons in the first hidden layer. If this is different from zero, it over writes the values in
            hidden_layer_sizes. It is useful for optimisation with Osprey.
        :hl2: int, default 0
            Number of neurons in the second hidden layer. If this is different from zero, it over writes the values in
            hidden_layer_sizes. It is useful for optimisation with Osprey.
        :hl3: int, default 0
            Number of neurons in the third hidden layer. If this is different from zero, it over writes the values in
            hidden_layer_sizes. It is useful for optimisation with Osprey.
        :descriptor: string, default "inverse_dist"
            This determines the choice of descriptor to be used as the input to the neural net. The current
            possibilities are:
            "inverse_dist"
        """

        # Initialising the parameters
        self.alpha_reg = alpha_reg
        self.alpha_grad = alpha_grad
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        # To make this work with Osprey
        if hl1 == 0 and hl2 == 0 and hl3 == 0:
            self.hidden_layer_sizes = hidden_layer_sizes
            if any(l == 0 for l in self.hidden_layer_sizes):
                raise ValueError("You have a hidden layer with 0 neurons in it.")

        else:
            self.hidden_layer_sizes = (hl1, hl2, hl3)
            if any(l == 0 for l in self.hidden_layer_sizes):
                raise ValueError("You have a hidden layer with 0 neurons in it.")


        # Other useful parameters
        self.alreadyInitialised = False
        self.loadedModel = False
        self.trainCost = []
        self.testCost = []
        self.isVisReady = False

        # Available descriptors
        self.available_descriptors = {
            "inverse_dist": inv.inv_dist
        }
        self.descriptor = self.available_descriptors[descriptor]

    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.

        :X: array of shape (n_samples, n_features).

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples, n_outputs).

            This contains the target values for each sample in the X matrix.

        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)

        self.alreadyInitialised = True

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
            X_des = self.descriptor(in_data, n_atoms=self.n_atoms)

        # Number of features in the descriptor
        self.n_feat = int(self.n_atoms * (self.n_atoms - 1) * 0.5)

        # Randomly initialisation of the weights and biases
        with tf.name_scope('weights'):
            weights, biases = self.__generate_weights(n_out=(1+3*self.n_atoms))

            # Log weights for tensorboard
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

        # Initialisation of the variables
        init = tf.global_variables_initializer()
        merged_summary = tf.summary.merge_all()

        # Running the graph
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(logdir="/Users/walfits/Repositories/Aglaia/tensorboard",graph=sess.graph)
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

                summary = sess.run(merged_summary, feed_dict={in_data:X})
                summary_writer.add_summary(summary, iter)

                self.trainCost.append(avg_cost)

            # Saving the weights for later re-use
            self.all_weights = []
            self.all_biases = []
            for ii in range(len(weights)):
                self.all_weights.append(sess.run(weights[ii]))
                self.all_biases.append(sess.run(biases[ii]))

    def save_NN(self, dir):

        if self.alreadyInitialised == False:
            raise Exception("The fit function has not been called yet, so the model has not been trained yet.")

        # Making the placeholder for the data
        xyz_test = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")

        # Making the descriptor from the Cartesian coordinates
        X_des = self.descriptor(xyz_test, n_atoms=self.n_atoms)

        # Setting up the trained weights
        weights = []
        biases = []

        for ii in range(len(self.all_weights)):
            weights.append(tf.Variable(self.all_weights[ii], name="weights_restore"))
            biases.append(tf.Variable(self.all_biases[ii], name="biases_restore"))

        # Calculating the ouputs
        out_NN = self.modelNN(X_des, weights, biases)

        init = tf.global_variables_initializer()

        # Object needed to save the model
        all_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            # Saving the graph
            all_saver.save(sess, dir+"/")

    def load_NN(self, dir):

        # Inserting the weights into the model
        with tf.Session() as sess:
            # Loading a saved graph
            file = dir + "/.meta"
            saver = tf.train.import_meta_graph(file)
            saver.restore(sess, tf.train.latest_checkpoint(dir))

            # The model is loaded in the default graph
            graph = tf.get_default_graph()

            # Loading the graph of out_NN
            self.out_NN = graph.get_tensor_by_name("output_node_1:0")
            self.in_data = graph.get_tensor_by_name("Cartesian_coord_1:0")


            saver.restore(sess, tf.train.latest_checkpoint(dir+"/"))
            sess.run(tf.global_variables_initializer())

        self.loadedModel = True

    def predict(self, X):
        """
        This function uses the X data and plugs it into the model and then returns the predicted y.

        :X: array of shape (n_samples, n_features)
            This contains the input data with samples in the rows and features in the columns.

        :return: array of size (n_samples, n_outputs)
            This contains the predictions for the target values corresponding to the samples contained in X.
        """

        check_array(X)

        if self.alreadyInitialised:

            # MAking the placeholder for the data
            xyz_test = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")

            # Making the descriptor from the Cartesian coordinates
            X_des = self.descriptor(xyz_test, n_atoms=self.n_atoms)

            # Setting up the trained weights
            weights = []
            biases = []

            for ii in range(len(self.all_weights)):
                weights.append(tf.Variable(self.all_weights[ii]))
                biases.append(tf.Variable(self.all_biases[ii]))

            # Calculating the ouputs
            out_NN = self.modelNN(X_des, weights, biases)

            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                ene_forces_NN = sess.run(out_NN, feed_dict={xyz_test: X})

            return ene_forces_NN

        elif self.loadedModel:
            feed_dic = {self.in_data: X}

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                ene_forces_NN = sess.run(self.out_NN, feed_dict=feed_dic)

            return ene_forces_NN

        else:
            raise Exception("The fit function has not been called yet, so the model has not been trained yet.")

    def modelNN(self, X, weights, biases):
        """
        This function evaluates the output of the neural network. It takes as input a data set, the weights and the
        biases.

        :X: tf.placeholder of shape (n_samples, n_features)
        :weights: list of tf.Variables of length len(hidden_layer_sizes) + 1
        :biases: list of tf.Variables of length len(hidden_layer_sizes) + 1
        :return: tf.Variable of size (n_samples, n_outputs)
        """

        # Calculating the activation of the first hidden layer
        z = tf.add(tf.matmul(X, tf.transpose(weights[0])), biases[0])
        h = tf.nn.sigmoid(z)

        # Calculating the activation of all the hidden layers
        for ii in range(len(self.hidden_layer_sizes)-1):
            z = tf.add(tf.matmul(h, tf.transpose(weights[ii+1])), biases[ii+1])
            h = tf.nn.sigmoid(z)

        # Calculating the output of the last layer
        z = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1], name="output_node")

        return z

    def __generate_weights(self, n_out):
        """
        This function generates the weights and the biases. It does so by looking at the size of the hidden layers,
        the number of features in the descriptor and the number of outputs. The weights are initialised randomly.

        :n_out: int, number of outputs
        :return: two lists (of length n_hidden_layers + 1) of tensorflow variables
        """

        weights = []
        biases = []

        # Weights from input layer to first hidden layer
        weights.append(tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[0], self.n_feat], stddev=0.1),
                                   name='weight_in'))
        biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[0]]), name='bias_in'))

        # Weights from one hidden layer to the next
        for ii in range(len(self.hidden_layer_sizes) - 1):
            weights.append(tf.Variable(
                tf.truncated_normal([self.hidden_layer_sizes[ii + 1], self.hidden_layer_sizes[ii]], stddev=0.1),
                name='weight_hidden'))
            biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[ii + 1]]), name='bias_hidden'))

        # Weights from lat hidden layer to output layer
        weights.append(tf.Variable(tf.truncated_normal([n_out, self.hidden_layer_sizes[-1]], stddev=0.1), name='weight_out'))
        biases.append(tf.Variable(tf.zeros([n_out]), name='bias_out'))

        return weights, biases

    def __reg_term(self, weights):
        """
        This function calculates the regularisation term to the cost function.

        :weights: list of tensorflow tensors
        :return: tensorflow scalar
        """

        reg_term = tf.zeros(shape=1, name="regu_term")

        for i in range(len(weights)):
            reg_term = reg_term + tf.reduce_sum(tf.square(weights[i]))

        return reg_term

    def checkBatchSize(self):
        """
        This function is called to check if the batch size has to take the default value or a user-set value.
        If it is a user set value, it checks whether it is a reasonable value.

        :return: int

            The default is 100 or to the total number of samples present if this is smaller than 100. Otherwise it is
            checked whether it is smaller than 1 or larger than the total number of samples.
        """
        if self.batch_size == 'auto':
            batch_size = min(100, self.n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > self.n_samples:
                print("Warning: Got `batch_size` less than 1 or larger than sample size. It is going to be clipped")
                batch_size = np.clip(self.batch_size, 1, self.n_samples)
            else:
                batch_size = self.batch_size

        return batch_size

    def score_new(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels. It calculates the R^2 value. It is used during the
        training of the model.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples, n_outputs)

            This contains the target values for each sample in the X matrix.

        :sample_weight: array of shape (n_samples,)

            Sample weights (not sure what this is, but i need it for inheritance from the BaseEstimator)

        :return: double
            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.
        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

    def plot_cost(self):
        """
        This function plots the value of the cost function as a function of the iterations. It is used in the fitting.
        """
        df = pd.DataFrame()
        df["Iterations"] = range(len(self.trainCost))
        df["Cost"] = self.trainCost
        lm = sns.lmplot('Iterations', 'Cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)
        # lm.set(yscale="log")
        plt.show()

    def correlation_plot(self, y_nn, y_true):
        """
        This function plots a correlation plot.

        :y_nn: list of values predicted by the neural net
        :y_true: list of gorund truth values
        """
        df = pd.DataFrame()
        df["NN_prediction"] = y_nn
        df["True_value"] = y_true
        lm = sns.lmplot('True_value', 'NN_prediction', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
        # lm.set(ylim=[-648.20*627.51, -648.15*627.51])
        # lm.set(xlim=[-648.20*627.51, -648.15*627.51])
        plt.show()

