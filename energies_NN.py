"""
This module implements a Tensorflow neural network (with scalar output) as a Scikit learn estimator.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import os

class MLPRegFlow(BaseEstimator, ClassifierMixin):
    """
    Neural-network with multiple hidden layers to do regression.
    This model optimises the squared error function using the Adam optimiser.


    :hidden_layer_sizes: Tuple, length = number of hidden layers, default (5,).

        The ith element represents the number of neurons in the ith
        hidden layer.

    :alpha: float, default 0.0001

        L2 penalty (regularization term) parameter.

    :batch_size: int, default 'auto'.

        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`

    :learning_rate_init: double, default 0.001.

        The value of the learning rate in the numerical minimisation.

    :max_iter: int, default 200.

        Total number of iterations that will be carried out during the training process.

    :hl1: int, default = 0

        These are the number of neurons in a particular hidden layer. hl1 is the number of neurons in the first hidden
        layer, hl2 is the number of neurons in the second hidden layer, etc. This is present so that one can change the
        number of neurons in Osprey. Osprey can only pass ints or floats to an estimator (not a tuple). Therefore the
        variable hidden_layer_sizes has to be constructed inside the estimator. If you want more than 3 hidden layers
        add hl4, hl5 ...

    :tensorboard: bool, default False

        A flag that lets you decide whether to save things to tensorboard or not

    :print_step: int, default 200

        Tells how many time to run the summaries to tensorboard.

    """

    def __init__(self, hidden_layer_sizes=(5,), alpha=0.0001, batch_size='auto', learning_rate_init=0.001,
                 max_iter=80, hl1=0, hl2=0, hl3=0, tensorboard=False, print_step = 200):

        # Initialising the parameters
        self.alpha = alpha
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


        # Initialising parameters needed for the Tensorflow part
        self.alreadyInitialised = False
        self.trainCost = []
        self.testCost = []
        self.isVisReady = False
        self.tensorboard = tensorboard
        self.print_step = print_step

    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.

        :X: array of shape (n_samples, n_features).

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,).

            This contains the target values for each sample in the X matrix.
        """

        # Creating a tensorboard directory if logging to tensorboard
        if self.tensorboard:
            self.board_dir = os.getcwd() + "/tensorboard"
            if not os.path.exists(self.board_dir):
                os.makedirs(self.board_dir)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Modification of the y data, because tensorflow wants a column vector, while scikit learn uses a row vector
        y = np.reshape(y, (len(y), 1))

        # Number of features and number of samples in the training set
        self.n_feat = X.shape[1]
        self.n_samples = X.shape[0]

        # Check the value of the batch size
        self.batch_size = self.checkBatchSize()

        # Initial set up of the NN
        with tf.name_scope("data"):
            X_train = tf.placeholder(tf.float32, [None, self.n_feat], name="Descriptor")
            Y_train = tf.placeholder(tf.float32, [None, 1], name="Energies")

        # This part either randomly initialises the weights and biases or restarts training from wherever it was stopped
        with tf.name_scope("weights"):
            weights, biases = self.__generate_weights()
            self.alreadyInitialised = True

            # Log weights for tensorboard
            if self.tensorboard:
                tf.summary.histogram("weights_in", weights[0])
                for ii in range(len(self.hidden_layer_sizes) - 1):
                    tf.summary.histogram("weights_hidden", weights[ii + 1])
                tf.summary.histogram("weights_out", weights[-1])

        with tf.name_scope("model"):
            model = self.modelNN(X_train, weights, biases)

        with tf.name_scope("cost_func"):
            cost = self.costReg(model, Y_train, weights, self.alpha)

        if self.tensorboard:
            cost_summary = tf.summary.scalar('cost', cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)

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
                summary_writer = tf.summary.FileWriter(logdir=self.board_dir, graph=sess.graph)
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

            # Saving the weights for later re-use
            self.all_weights = []
            self.all_biases = []
            for ii in range(len(weights)):
                self.all_weights.append(sess.run(weights[ii]))
                self.all_biases.append(sess.run(biases[ii]))

    def modelNN(self, X, weights, biases):
        """
        This function evaluates the output of the neural network. It takes as input a data set, the weights and the
        biases.

        :X: tf.placeholder of shape (n_samples, n_features)
        :weights: list of tf.Variables of length len(hidden_layer_sizes) + 1
        :biases: list of tf.Variables of length len(hidden_layer_sizes) + 1
        :return: tf.Variable of size (n_samples, 1)
        """

        # Calculating the activation of the first hidden layer
        z = tf.add(tf.matmul(X, tf.transpose(weights[0])), biases[0])
        h = tf.nn.sigmoid(z)

        # Calculating the activation of all the hidden layers
        for ii in range(len(self.hidden_layer_sizes)-1):
            z = tf.add(tf.matmul(h, tf.transpose(weights[ii+1])), biases[ii+1])
            h = tf.nn.sigmoid(z)

        # Calculating the output of the last layer
        z = tf.add(tf.matmul(h, tf.transpose(weights[-1])), biases[-1])

        return z

    def __generate_weights(self):
        """
        This function generates the weights and the biases. It does so by looking at the size of the hidden layers and
        the number of features in the descriptor. The weights are initialised randomly.

        :return: lists (of length n_hidden_layers + 1) of tensorflow variables
        """

        weights = []
        biases = []

        # Weights from input layer to first hidden layer
        weights.append(tf.Variable(tf.truncated_normal([self.hidden_layer_sizes[0], self.n_feat], stddev=0.01),
                                   name='weight_in'))
        biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[0]]), name='bias_in'))

        # Weights from one hidden layer to the next
        for ii in range(len(self.hidden_layer_sizes) - 1):
            weights.append(tf.Variable(
                tf.truncated_normal([self.hidden_layer_sizes[ii + 1], self.hidden_layer_sizes[ii]], stddev=0.01),
                name='weight_hidden'))
            biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[ii + 1]]), name='bias_hidden'))

        # Weights from lat hidden layer to output layer
        weights.append(
            tf.Variable(tf.truncated_normal([1, self.hidden_layer_sizes[-1]], stddev=0.01), name='weight_out'))
        biases.append(tf.Variable(tf.zeros([1]), name='bias_out'))

        return weights, biases

    def costReg(self, qm_data, nn_data, weights, regu):
        """
        This function calculates the squared error cost function with L2 regularisation.

        :nn_data: tensor

            This tensor contains the neural network model.

        :qm_data: TensorFlow Place holder

            This tensor contains the y part of the data once the graph is initialised.

        :weights: array of TensorFlow variables of shape (len(hidden_layer_sizes+1), )

            It contains the weights for each hidden layer and the output layer.

        :regu: float

            The parameter that tunes the amount of regularisation.

        :return: tensor
        """
        err = tf.square(tf.subtract(qm_data, nn_data))
        cost = tf.reduce_mean(err, name="unreg_cost") # scalar

        # Calculating the regularisation term
        reg_term = tf.Variable(tf.zeros([]), name="regu_term")
        for i in range(len(weights)):
            reg_term = reg_term + tf.reduce_sum(tf.square(weights[i]))


        # Multiplying the regularisation term by the regularisation parameter
        reg_l2 = tf.scalar_mul(regu, reg_term)

        # Regularised cost
        cost_reg = tf.add(cost, reg_l2, name="reg_cost")

        return cost_reg

    def plotLearningCurve(self):
        """
        This function plots the cost versus the number of iterations for the training set and the test set in the
        same plot. The cost on the train set is calculated every 50 iterations.
        """

        df = pd.DataFrame()
        df["Iterations"] = range(len(self.trainCost))
        df["Cost"] = self.trainCost
        sns.set()
        lm = sns.lmplot('Iterations', 'Cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5},
                        fit_reg=False)
        # lm.set(yscale="log")
        plt.show()

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

    def checkIsFitted(self):
        """
        This function checks whether the weights and biases have been changed from their initial values.

        :return: True if the weights and biases are not all zero.
        """
        if self.alreadyInitialised == False:
            raise Exception("The fit function has not been called yet")
        else:
            return True

    def predict(self, X):
        """
        This function uses the X data and plugs it into the model and then returns the predicted y

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :return: array of size (n_samples,)

            This contains the predictions for the target values corresponding to the samples contained in X.

        """

        check_array(X)

        if self.alreadyInitialised:

            X_test = tf.placeholder(tf.float32, [None, self.n_feat], name="descriptor")

            weights = []
            biases = []

            for ii in range(len(self.all_weights)):
                weights.append(tf.Variable(self.all_weights[ii]))
                biases.append(tf.Variable(self.all_biases[ii]))

            model = self.modelNN(X_test, weights, biases)

            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                predictions = sess.run(model, feed_dict={X_test: X})
                predictions = np.reshape(predictions,(predictions.shape[0],))

            return predictions
        else:
            raise Exception("The fit function has not been called yet, so the model has not been trained yet.")

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels. It calculates the R^2 value. It is used during the
        training of the model.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :sample_weight: array of shape (n_samples,)

            Sample weights (not sure what this is, but i need it for inheritance from the BaseEstimator)

        :return: double
            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.
        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

    def scoreFull(self, X, y):
        """
        This scores the predictions more thouroughly than the function 'score'. It calculates the r2, the root mean
        square error, the mean absolute error and the largest positive/negative outliers. They are all in the units of
        the data passed.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :return:
        :r2: double

            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.

        :rmse: double

            This is the root mean square error

        :mae: double

            This is the mean absolute error

        :lpo: double

            This is the largest positive outlier.

        :lno: double

            This is the largest negative outlier.

        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        lpo, lno = self.largestOutliers(y, y_pred)

        return r2, rmse, mae, lpo, lno

    def largestOutliers(self, y_true, y_pred):
        """
        This function calculates the larges positive and negative outliers from the predictions of the neural net.

        :y_true: array of shape (n_samples,)

            This contains the target values for each sample.

        :y_pred: array of shape (n_samples,)

            This contains the neural network predictions of the target values for each sample.

        :return:

        :lpo: double

            This is the largest positive outlier.

        :lno: double

            This is the largest negative outlier.
        """
        diff = y_pred - y_true
        lpo = np.amax(diff)
        lno = - np.amin(diff)

        return lpo, lno

    def errorDistribution(self, X, y):
        """
        This function plots histograms of how many predictions have an error in a certain range.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.
        """
        y_pred = self.predict(X)
        diff_kJmol = (y - y_pred)*2625.50
        df = pd.Series(diff_kJmol, name="Error (kJ/mol)")
        # sns.set_style(style='white')
        # sns.distplot(df, color="#f1ad1e")
        # sns.plt.savefig("ErrorDist.png", transparent=True, dpi=800)
        plt.show()

    def correlationPlot(self, X, y, ylim=(1.90, 1.78), xlim=(1.90, 1.78)):
        """
        This function plots a correlation plot of the values that are in the data set and the NN predictions. It expects
        the target values to be in Hartrees.

        :X: array of shape (n_samples, n_features)

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :ylim: tuple of shape (2,) containing doubles

            These are the limits of the y values for the plot.

        :xlim: tuple of shape (2,) containing doubles

            These are the limits of the x values for the plot.
        """
        y_pred = self.predict(X)
        df = pd.DataFrame()
        df['High level calculated energies (Ha)'] = y
        df['NN predicted energies (Ha)'] = y_pred
        lm = sns.lmplot('High level calculated energies (Ha)', 'NN predicted energies (Ha)', data=df,
                        scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
        lm.set(ylim=ylim)
        lm.set(xlim=xlim)
        plt.show()

    def plotWeights(self):
        """
        This function plots the weights of the first layer of the neural network as a heat map.
        """

        w1_square_tot = []

        for i in range(self.hidden_layer_sizes[0]):
            w1_square = self.reshape_triang(self.w1[i], 7)
            w1_square_tot.append(w1_square)

        n = int(np.ceil(np.sqrt(self.hidden_layer_sizes)))
        additional = n**2 - self.hidden_layer_sizes[0]

        fig, axn = plt.subplots(n, n, sharex=True, sharey=True)
        fig.set_size_inches(11.7, 8.27)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i, ax in enumerate(axn.flat):
            if i >= self.hidden_layer_sizes[0]:
                break
            df = pd.DataFrame(w1_square_tot[i])
            sns.heatmap(df,
                        ax=ax,
                        cbar=i == 0,
                        vmin=-0.2, vmax=0.2,
                        cbar_ax=None if i else cbar_ax, cmap="PiYG")

        fig.tight_layout(rect=[0, 0, 0.9, 1])
        # sns.plt.savefig("weights_l1.png", transparent=False, dpi=600)
        # sns.plt.show()

    def reshape_triang(self, X, dim):
        """
        This function reshapes a single flattened triangular matrix back to a square diagonal matrix.

        :X: array of shape (n_atoms*(n_atoms+1)/2, )

            This contains a sample of the Coulomb matrix trimmed down so that it contains only the a triangular matrix.

        :dim: int

            The triangular matrix X will be reshaped to a matrix that has size dim by dim.


        :return: array of shape (n_atoms, n_atoms)

            This contains the square diagonal matrix.
        """

        x_square = np.zeros((dim, dim))
        counter = 0
        for i in range(dim):
            for j in range(i, dim):
                x_square[i][j] = X[counter]
                x_square[j][i] = X[counter]
                counter = counter + 1

        return x_square

    def __vis_input(self, initial_guess):
        """
        This function does gradient ascent to generate an input that gives the highest activation for each neuron of
        the first hidden layer.

        :initial_guess: array of shape (n_features,)

            A coulomb matrix to use as the initial guess to the gradient ascent in the hope that the closest local
            maximum will be found.

        :return: list of arrays of shape (num_atoms, num_atoms)

            each numpy array is the input for a particular neuron that gives the highest activation.

        """

        self.isVisReady = True
        initial_guess = np.reshape(initial_guess, newshape=(1, initial_guess.shape[0]))
        input_x = tf.Variable(initial_guess, dtype=tf.float32)
        activations = []
        iterations = 7000
        lambda_reg = 0.0002
        self.x_square_tot = []

        for node in range(self.hidden_layer_sizes[0]):

            # Calculating the activation of the first layer
            w1_node = tf.constant(self.w1[node], shape=(1,self.n_feat))
            b1_node = tf.constant(self.b1[node])
            z1 = tf.add(tf.matmul(tf.abs(input_x), tf.transpose(w1_node)), b1_node)
            a1 = tf.nn.sigmoid(z1)
            a1_reg = a1 - lambda_reg * tf.tensordot(input_x, tf.transpose(input_x), axes=1)

            # Function to maximise a1
            optimiser = tf.train.AdamOptimizer(learning_rate=0.01).minimize(-a1_reg)

            # Initialising the model
            init = tf.global_variables_initializer()


            # Running the graph
            with tf.Session() as sess:
                sess.run(init)

                for i in range(iterations):
                    sess.run(optimiser)

                temp_a1 = sess.run(a1)
                activations.append(temp_a1)     # Calculating the activation for checking later if a node has converged
                final_x = sess.run(input_x)     # Storing the best input

            x_square = self.reshape_triang(final_x[0,:], 7)
            self.x_square_tot.append(x_square)
        print("The activations at the end of the optimisations are:")
        # print(activations)

        return self.x_square_tot

    def vis_input_matrix(self, initial_guess, write_plot=False):
        """
        This function calculates the inputs that would give the highest activations of the neurons in the first hidden
        layer of the neural network. It then plots them as a heat map.

        :initial_guess: array of shape (n_features,)

            A coulomb matrix to use as the initial guess to the gradient ascent in the hope that the closest local
            maximum will be found.

        :write_plot: boolean, default False

            If this is true, the plot is written to a png file.
        """

        if self.isVisReady == False:
            self.x_square_tot = self.__vis_input(initial_guess)

        n = int(np.ceil(np.sqrt(self.hidden_layer_sizes)))
        additional = n ** 2 - self.hidden_layer_sizes[0]

        fig, axn = plt.subplots(n, n, sharex=True, sharey=True)
        fig.set_size_inches(11.7, 8.27)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        counter = 0

        for i, ax in enumerate(axn.flat):
            df = pd.DataFrame(self.x_square_tot[counter])
            ax.set(xticks=[], yticks=[])
            sns.heatmap(df, ax=ax, cbar=i == 0, cmap='RdYlGn',
                        vmax=8, vmin=-8,
                        cbar_ax=None if i else cbar_ax)
            counter = counter + 1
            if counter >= self.hidden_layer_sizes[0]:
                break

        fig.tight_layout(rect=[0, 0, 0.9, 1])
        if write_plot==True:
            sns.plt.savefig("high_a1_input.png", transparent=False, dpi=600)
        sns.plt.show()

    def vis_input_network(self, initial_guess, write_plot=False):
        """
        This function calculates the inputs that would give the highest activations of the neurons in the first hidden
        layer of the neural network. It then plots them as a netwrok graph.

        :initial_guess: array of shape (n_features,)

            A coulomb matrix to use as the initial guess to the gradient ascent in the hope that the closest local
            maximum will be found.

        :write_plot: boolean, default False

            If this is true, the plot is written to a png file.
        """
        import networkx as nx

        if self.isVisReady == False:
            self.x_square_tot = self.__vis_input(initial_guess)

        n = int(np.ceil(np.sqrt(self.hidden_layer_sizes)))

        fig = plt.figure(figsize=(10, 8))
        for i in range(n**2):
            if i >= self.hidden_layer_sizes[0]:
                break
            fig.add_subplot(n,n,1+i)
            A = np.matrix(self.x_square_tot[i])
            graph2 = nx.from_numpy_matrix(A, parallel_edges=False)
            # nodes and their label
            # pos = {0: np.array([0.46887886, 0.06939788]), 1: np.array([0, 0.26694294]),
            #        2: np.array([0.3, 0.56225267]),
            #        3: np.array([0.13972517, 0.]), 4: np.array([0.6, 0.9]), 5: np.array([0.27685853, 0.31976436]),
            #        6: np.array([0.72, 0.9])}
            pos = {}
            for i in range(7):
                x_point = 0.6*np.cos((i+1)*2*np.pi/7)
                y_point = 0.6*np.sin((i+1)*2*np.pi/7)
                pos[i] = np.array([x_point, y_point])
            labels = {}
            labels[0] = 'H'
            labels[1] = 'H'
            labels[2] = 'H'
            labels[3] = 'H'
            labels[4] = 'C'
            labels[5] = 'C'
            labels[6] = 'N'
            node_size = np.zeros(7)
            for i in range(7):
                node_size[i] =  abs(graph2[i][i]['weight'])*10
            nx.draw_networkx_nodes(graph2, pos, node_size=node_size)
            nx.draw_networkx_labels(graph2, pos, labels=labels, font_size=15, font_family='sans-serif', font_color='blue')
            # edges
            edgewidth = [d['weight'] for (u, v, d) in graph2.edges(data=True)]
            nx.draw_networkx_edges(graph2, pos, width=edgewidth)
            plt.axis('off')

        if write_plot==True:
            plt.savefig("high_a1_network.png")  # save as png

        plt.show()  # display



# This example tests the module on fitting a simple cubic function and then plots the results

if __name__ == "__main__":

    estimator = MLPRegFlow(hidden_layer_sizes=(5, 5, 5), learning_rate_init=0.01, max_iter=5000, alpha=0)
    x = np.arange(-2.0, 2.0, 0.05)
    X = np.reshape(x, (len(x), 1))
    y = np.reshape(X ** 3, (len(x),))

    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    #  Visualisation of predictions
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(x, y, label="original", marker="o", c="r")
    ax2.scatter(x, y_pred, label="predictions", marker="o", c='b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()

    # Correlation plot
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.scatter(y, y_pred, marker="o", c="r")
    ax3.set_xlabel('original y')
    ax3.set_ylabel('prediction y')
    plt.show()

    # estimator.errorDistribution(X, y)

