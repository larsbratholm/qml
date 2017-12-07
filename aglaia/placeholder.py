class Test(unittest.TestCase):

    def test_dist_mat(self):
        """
        This function tests the distance squared matrix function
        """

        xyz_test = tf.constant([[0., 0., 0., 1., 0., 0.], [0., 0., 0., 2., 0., 0.]])
        exp_result = [[1.0], [4.0]]
        mat = des.dist_mat(xyz_test, n_atoms=2)

        sess = tf.Session()
        actual_result = sess.run(mat)

        self.assertTrue(np.all(np.isclose(exp_result, actual_result)))

    def test_inv_mat(self):
        xyz_test = tf.constant([[0., 0., 0., 1., 0., 0.], [0., 0., 0., 2., 0., 0.]])
        exp_result = [[1.0], [0.25]]

        mat = des.inv_dist(xyz_test, n_atoms=2)

        sess = tf.Session()
        actual_result = sess.run(mat)

        self.assertTrue(np.all(np.isclose(exp_result, actual_result)))

    def test_loading_1(self):
        """
        This test checks whether a very simple tf graph can be saved and then loaded back.
        """
        import os

        # Creating a directory
        cwd = os.getcwd()
        save_dir = cwd + "/tmp_dir/"

        # Defining some variables
        v1 = tf.Variable(1., name="v1")
        v2 = tf.Variable(2., name="v2")

        # Defining an operation
        a = tf.add(v1, v2)

        # Creating the saver object
        all_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            all_saver.save(sess, save_dir)

        saver = tf.train.import_meta_graph("tmp_dir/.meta")

        graph = tf.get_default_graph()

        restore_tensor = graph.get_tensor_by_name("Add:0")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_result = sess.run(restore_tensor)
            expected_result = 3.0

        self.assertTrue(actual_result == expected_result)

    def test_loading_2(self):
        """
        This test checks whether a very simple tf graph can be saved and then loaded back.
        """
        import os

        # Creating a directory
        cwd = os.getcwd()
        save_dir = cwd + "/tmp_dir/"

        # Defining some variables
        v1 = tf.placeholder("float", name="v1")
        v2 = tf.placeholder("float", name="v2")
        feed_dic = {v1: 1.0, v2: 2.0}
        v3 = tf.Variable(2.0, name="v3")

        # Defining an operation
        a1 = tf.add(v1, v2, name="op1")
        a2 = tf.add(a1, v3, name="op2")

        # Creating the saver object
        all_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            all_saver.save(sess, save_dir)

        # Restoring

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("tmp_dir/.meta")
            saver.restore(sess, tf.train.latest_checkpoint('tmp_dir/'))

            graph = tf.get_default_graph()

            new_v1 = graph.get_tensor_by_name("v1:0")
            new_v2 = graph.get_tensor_by_name("v2:0")
            feed_dic = {new_v1: 1.0, new_v2: 2.0}
            restore_tensor = graph.get_tensor_by_name("op2:0")

            sess.run(tf.global_variables_initializer())
            actual_result = sess.run(restore_tensor, feed_dict=feed_dic)
            expected_result = 5.0

        self.assertTrue(actual_result == expected_result)

    def test_neural_net1(self):

        """
        This test checks whether the predictions of the model just after being trained are the same as after the model
        is reloaded.
        """

        X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [4, 5, 6, 7, 8, 9, 10, 11, 12]])

        y = np.array([[10, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                      [11, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [12, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [13, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [14, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

        estimator1 = nn.MLPRegFlow(max_iter=50)
        estimator1.fit(X, y)
        expected_y = estimator1.predict(X)
        estimator1.save_NN("/Users/walfits/Repositories/Aglaia/Examples/Models/")

        estimator2 = nn.MLPRegFlow(max_iter=50)
        estimator2.load_NN("/Users/walfits/Repositories/Aglaia/Examples/Models/")
        actual_y = estimator2.predict(X)

        self.assertTrue(np.all(np.isclose(expected_y, actual_y)))

    def test_gpu_access(self):
        """This test checks whether the GPUs can be found and used. Should only be run on a machine that has GPUs."""
        with tf.device('/gpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)


        with tf.Session() as sess:
            actual_result = sess.run(c)

        expected_result = np.array([[ 22.,  28.], [ 49.,  64.]])

        self.assertTrue(np.all(np.isclose(expected_result, actual_result)))

def dist_mat(X, n_atoms):
    """
    This function takes in a tensor containing all the cartesian coordinates of the atoms in a trajectory. Each line is
    a different configuration. It returns the upper triangular part of the distance squared matrix. (Note: couldnt use
    the distance instead of the distance squared, because the gradient of sqrt(X^2) approaches inf when X^2 goes to zero.
    This gives numerical instability when calculating gradients).

    :X: tensor with shape (n_samples, n_features)
    :n_samples: number of samples
    :n_atoms: number of atoms = int(n_features/3)
    :return: tensor of shape (n_samples, int(n_atoms * (n_atoms-1) * 0.5))
    """


    # This part generates the inverse matrix
    xyz_3d = tf.reshape(X, shape=(tf.shape(X)[0], n_atoms, 3))
    expanded_a = tf.expand_dims(xyz_3d, 2)
    expanded_b = tf.expand_dims(xyz_3d, 1)
    diff2 = tf.squared_difference(expanded_a, expanded_b, name='square_diff')
    diff2_sum = tf.reduce_sum(diff2, axis=3)
    # diff_sum = tf.sqrt(diff2_sum, name='diff')   # This has shape [n_samples, n_atoms, n_atoms]

    # This part takes the upper triangular part (no diagonal) of the distance matrix and flattens it to a vector
    ones = tf.ones_like(diff2_sum)
    mask_a = tf.matrix_band_part(ones, 0, -1)
    mask_b = tf.matrix_band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool, name='mask') # Transfoorm into bool

    upper_triangular_conc = tf.boolean_mask(diff2_sum, mask, name='descript_concat')
    upper_triangular = tf.reshape(upper_triangular_conc, shape=(tf.shape(X)[0], int(n_atoms * (n_atoms-1) * 0.5)), name='descript')

    return upper_triangular

def inv_dist(X, n_atoms):
    """
    This function calculates the inverse distance squared matrix.

    :X: tensor with shape (n_samples, n_features)
    :n_samples: number of samples
    :n_atoms: number of atoms = int(n_features/3)
    :return: tensor of shape (n_samples, int(n_atoms * (n_atoms-1) * 0.5))
    """

    dist_matrix = dist_mat(X, n_atoms=n_atoms)

    inv_dist_matrix = 1/dist_matrix

    return inv_dist_matrix

def load_data(directory, n_samples=50):
    # Making a list of the files to look at
    list_of_files = []
    subdirs = [x[0] for x in os.walk(directory)]
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]

    # Extracting the data from each file
    atoms = ""
    traj_coord = []
    ene = []
    forces = []

    for subdir in subdirs:
        for file in files[:n_samples]:
            filename = subdir + file
            with open(filename) as f:
                lines = f.readlines()
                tokens = lines[1].split(';')

                ene.append(float(tokens[0]))

                forces_mat = ast.literal_eval(tokens[1])
                forces_list = []
                for item in forces_mat:
                    for i in range(3):
                        forces_list.append(item[i])
                forces.append(forces_list)

                coord = []
                for line in lines[2:]:
                    tokens = line.split()
                    atoms += tokens[0]
                    coord_float = [float(i) for i in tokens[1:]]
                    for i in range(3):
                        coord.append(coord_float[i])

                traj_coord.append(coord)

    traj_coord = np.asarray(traj_coord)
    ene = np.asarray(ene)
    forces = np.asarray(forces)

    return traj_coord, ene, forces


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


#   fit forces
        ## Placeholders for the input/output data
        #with tf.name_scope('Data'):
        #    tf_input = tf.placeholder(tf.float32, [None, self.n_coord], name="Coordinates")
        #    tf_output= tf.placeholder(tf.float32, [None, self.n_coord + 1], name="Energy_forces")

        ## Making the descriptor from the Cartesian coordinates
        #with tf.name_scope('Descriptor'):
        #    X_des = self.available_descriptors[self.descriptor](in_data, n_atoms=self.n_atoms)

        ## Number of features in the descriptor
        #self.n_features = int(self.n_atoms * (self.n_atoms - 1) * 0.5)

        ## Randomly initialisation of the weights and biases
        #with tf.name_scope('weights'):
        #    weights, biases = self.__generate_weights(n_out=(1+3*self.n_atoms))

        #    # Log weights for tensorboard
        #    if self.tensorboard:
        #        tf.summary.histogram("weights_in", weights[0])
        #        for ii in range(len(self.hidden_layer_sizes) - 1):
        #            tf.summary.histogram("weights_hidden", weights[ii + 1])
        #        tf.summary.histogram("weights_out", weights[-1])


        ## Calculating the output of the neural net
        #with tf.name_scope('model'):
        #    out_NN = self.modelNN(X_des, weights, biases)

        ## Obtaining the derivative of the neural net energy wrt cartesian coordinates
        #with tf.name_scope('grad_ene'):
        #    ene_NN = tf.slice(out_NN,begin=[0,0], size=[-1,1], name='ene_NN')
        #    grad_ene_NN = tf.gradients(ene_NN, in_data, name='dEne_dr')[0] * (-1)

        ## Calculating the cost function
        #with tf.name_scope('cost_funct'):
        #    err_ene_force = tf.square(tf.subtract(out_NN, out_data), name='err2_ene_force')
        #    err_grad = tf.square(tf.subtract(tf.slice(out_data, begin=[0,1], size=[-1,-1]), grad_ene_NN), name='err2_grad')

        #    cost_ene_force = tf.reduce_mean(err_ene_force, name='cost_ene_force')
        #    cost_grad = tf.reduce_mean(err_grad, name='cost_grad')

        #    reg_term = self.__reg_term(weights)

        #    cost = cost_ene_force + self.alpha_grad*cost_grad + self.alpha_reg * reg_term

        ## Training the network
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)

        #if self.tensorboard:
        #    cost_summary = tf.summary.scalar('cost', cost)

        ## Initialisation of the variables
        #init = tf.global_variables_initializer()
        #if self.tensorboard:
        #    merged_summary = tf.summary.merge_all()
        #    options = tf.RunOptions()
        #    options.output_partition_graphs = True
        #    options.trace_level = tf.RunOptions.SOFTWARE_TRACE
        #    run_metadata = tf.RunMetadata()

        ## Running the graph
        #with tf.Session() as sess:
        #    if self.tensorboard:
        #        summary_writer = tf.summary.FileWriter(logdir=self.board_dir,graph=sess.graph)
        #    sess.run(init)

        #    for iter in range(self.max_iter):
        #        # This is the total number of batches in which the training set is divided
        #        n_batches = int(self.n_samples / self.batch_size)
        #        # This will be used to calculate the average cost per iteration
        #        avg_cost = 0
        #        # Learning over the batches of data
        #        for i in range(n_batches):
        #            batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
        #            batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
        #            opt, c = sess.run([optimizer, cost], feed_dict={in_data: batch_x, out_data: batch_y})
        #            avg_cost += c / n_batches

        #            if self.tensorboard:
        #                if iter % self.print_step == 0:
        #                    # The options flag is needed to obtain profiling information
        #                    summary = sess.run(merged_summary, feed_dict={in_data: batch_x, out_data: batch_y}, options=options, run_metadata=run_metadata)
        #                    summary_writer.add_summary(summary, iter)
        #                    summary_writer.add_run_metadata(run_metadata, 'iteration %d batch %d' % (iter, i))

        #        self.trainCost.append(avg_cost)

        #    # Saving the weights for later re-use
        #    self.all_weights = []
        #    self.all_biases = []
        #    for ii in range(len(weights)):
        #        self.all_weights.append(sess.run(weights[ii]))
        #        self.all_biases.append(sess.run(biases[ii]))



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

#    def fit(self, X, y):
#        """
#        Fit the to data matrix X and target y.
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
