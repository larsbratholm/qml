import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import tensorflow as tf
import inverse_dist as inv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class MLPRegFlow(BaseEstimator, ClassifierMixin):


    def __init__(self, hidden_layer_sizes=(5,), alpha_reg=0.0001, alpha_grad=0.05, alpha_force=0.9, batch_size='auto', learning_rate_init=0.001,
                 max_iter=80, hl1=0, hl2=0, hl3=0):
        """
        Neural-network with multiple hidden layers to do regression.
        This model optimises the squared error function using the Adam optimiser.
        :hidden_layer_sizes: Tuple, length = number of hidden layers, default (5,).
            The ith element represents the number of neurons in the ith
            hidden layer.
        :alpha_reg: float, default 0.0001
            L2 penalty (regularization term) parameter.
        :alpha_grad: default 0.05
            Parameter that enables to tweak the importance of the gradient term in the cost function.
        :alpha_force: default 0.9
            Parameter that enables to tweak the importance of the force term in the cost function.
        :batch_size: int, default 'auto'.
            Size of minibatches for stochastic optimizers.
            If the solver is 'lbfgs', the classifier will not use minibatch.
            When set to "auto", `batch_size=min(200, n_samples)`
        :learning_rate_init: double, default 0.001.
            The value of the learning rate in the numerical minimisation.
        :max_iter: int, default 200.
            Total number of iterations that will be carried out during the training process.
        """

        # Initialising the parameters
        self.alpha_reg = alpha_reg
        self.alpha_grad = alpha_grad
        self.alpha_force = alpha_force
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

        # Available descriptors
        self.descriptors = {
            "inverse_dist": inv.inv_dist
        }


    def fit(self, X, y, descriptor="inverse_dist"):
        """
        Fit the model to data matrix X and target y.

        :X: array of shape (n_samples, n_features).

            This contains the input data with samples in the rows and features in the columns.

        :y: array of shape (n_samples,).

            This contains the target values for each sample in the X matrix.

        """

        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)

        self.alreadyInitialised = True

        # Modification of the y data, because tensorflow wants a column vector, while scikit learn uses a row vector

        self.n_coord = X.shape[1]
        self.n_samples = X.shape[0]
        self.n_atoms = int(X.shape[1]/3)

        # Check the value of the batch size
        self.batch_size = self.checkBatchSize()

        # Place holders for the data
        with tf.name_scope('Input'):
            in_data = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")
            out_data = tf.placeholder(tf.float32, [None, 3 * self.n_atoms + 1], name="Energy")

        # Making the descriptor from the Cartesian coordinates
        with tf.name_scope('Descriptor'):
            X_des = self.descriptors[descriptor](in_data, n_samples=self.batch_size, n_atoms=self.n_atoms)

        self.n_feat = int(self.n_atoms * (self.n_atoms - 1) * 0.5)

        # Randomly initialisation of the weights and biases

        with tf.name_scope('weights'):
            weights, biases = self.__generate_weights(n_out=(1+3*self.n_atoms))

            tf.summary.histogram("weights_in", weights[0])
            for ii in range(len(self.hidden_layer_sizes) - 1):
                tf.summary.histogram("weights_hidden", weights[ii + 1])
            tf.summary.histogram("weights_out", weights[-1])


        # Calculating the output of the neural net
        with tf.name_scope('model_ene_force'):
            out_NN = self.modelNN(X_des, weights, biases)


        # Obtaining the derivative of the neural net energy wrt cartesian coordinates
        with tf.name_scope('grad_ene'):
            ene_NN = tf.slice(out_NN,begin=[0,0], size=[-1,1], name='ene_NN')
            grad_ene_NN = tf.gradients(ene_NN, in_data, name='dEne_dr')[0] * (-1)

        with tf.name_scope('cost_funct'):
            # Calculating the cost function
            err_ene_force = tf.square(tf.subtract(out_NN, out_data), name='err2_ene_force')
            err_grad = tf.square(tf.subtract(tf.slice(out_data ,begin=[0,1], size=[-1,-1]), grad_ene_NN), name='err2_grad')

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



    def predict(self, X, descriptor="inverse_dist"):
        """
        This function uses the X data and plugs it into the model and then returns the predicted y
        :X: array of shape (n_samples, n_features)
            This contains the input data with samples in the rows and features in the columns.
        :return: array of size (n_samples,)
            This contains the predictions for the target values corresponding to the samples contained in X.
        """

        if self.alreadyInitialised:
            check_array(X)

            n_samples = X.shape[0]

            # MAking the placeholder for the data
            xyz_test = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")

            # Making the descriptor from the Cartesian coordinates
            X_des = self.descriptors[descriptor](xyz_test, n_samples=n_samples, n_atoms=self.n_atoms)

            # Setting up the trained weights
            weights = []
            biases = []

            for ii in range(len(self.all_weights)):
                weights.append(tf.Variable(self.all_weights[ii]))
                biases.append(tf.Variable(self.all_biases[ii]))


            # Calculating the ouputs
            out_NN = self.modelNN(X_des, weights, biases)
            ene_NN = tf.slice(out_NN,begin=[0,0], size=[-1,1], name='ene_NN')
            force_NN = tf.slice(out_NN,begin=[0,1], size=[-1,-1], name='force_NN')

            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                ene_pred = sess.run(ene_NN, feed_dict={xyz_test: X})
                force_pred = sess.run(force_NN, feed_dict={xyz_test: X})
                ene_pred = np.reshape(ene_pred, (ene_pred.shape[0],))

            return ene_pred, force_pred
        else:
            raise Exception("The fit function has not been called yet, so the model has not been trained yet.")

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

    def __generate_weights(self, n_out):
        """
        This function generates the weights and the biases. It does so by looking at the size of the hidden layers and
        the number of features in the descriptor. The weights are initialised randomly.

        :return: lists (of length n_hidden_layers + 1) of tensorflow variables
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

        :weights_ene: list of tensorflow tensors
        :param weights_force: list of tensorflow tensors
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

        :y: array of shape (n_samples,)

            This contains the target values for each sample in the X matrix.

        :sample_weight: array of shape (n_samples,)

            Sample weights (not sure what this is, but i need it for inheritance from the BaseEstimator)

        :return: double
            This is a score between -inf and 1 (best value is 1) that tells how good the correlation plot is.
        """
        ene = y[:,0]
        force = y[:, 1:]
        y_pred, dy_pred = self.predict(X)
        r2_ene = r2_score(ene, y_pred)
        r2_force = r2_score(force, dy_pred)
        r2 = (r2_ene + r2_force)/2
        return r2

    def plot_cost(self):
        df = pd.DataFrame()
        df["Iterations"] = range(len(self.trainCost))
        df["Cost"] = self.trainCost
        lm = sns.lmplot('Iterations', 'Cost', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5}, fit_reg=False)
        # lm.set(yscale="log")
        plt.show()

    def correlation_plot(self, y_nn, y_true):
        df = pd.DataFrame()
        df["NN_prediction"] = y_nn
        df["True_value"] = y_true
        lm = sns.lmplot('True_value', 'NN_prediction', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha": 0.5})
        # lm.set(ylim=[-648.20*627.51, -648.15*627.51])
        # lm.set(xlim=[-648.20*627.51, -648.15*627.51])
        plt.show()


if __name__ == "__main__":
    import extract

    coord_xyz, ene, forces = extract.load_data("/Users/walfits/Documents/aspirin/", n_samples=50)

    mean_ene = np.mean(ene)
    std_ene = np.std(ene)

    ene = (ene-mean_ene)/std_ene
    forces = forces/std_ene

    ene = np.reshape(ene,(ene.shape[0], 1))
    ene_force = np.concatenate((ene, forces), axis=1)

    #Hartree
    # ene = ene * 0.0015936
    # forces = forces * 0.0015936

    estimator = MLPRegFlow()
    estimator = MLPRegFlow(max_iter=5000, learning_rate_init=0.002, hidden_layer_sizes=(100,), batch_size=50,
                              alpha_reg=0.0, alpha_force=0.5, alpha_grad=1)
    estimator.fit(coord_xyz, ene_force)
    estimator.plot_cost()

    ene_pred, force_pred = estimator.predict(coord_xyz)
    estimator.correlation_plot(ene_pred, ene)
    print(estimator.score_new(coord_xyz, ene_force))
