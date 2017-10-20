import unittest
import inverse_dist as des
import Neural_net_2 as nn
import tensorflow as tf
import numpy as np

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
        estimator1.save_NN("/Users/walfits/Repositories/Aglaia/Examples/tmp_dir")

        estimator2 = nn.MLPRegFlow(max_iter=50)
        estimator2.load_NN("/Users/walfits/Repositories/Aglaia/Examples/tmp_dir")
        actual_y = estimator2.predict(X)

        self.assertTrue(np.all(np.isclose(expected_y, actual_y)))


if __name__ == "__main__":
    unittest.main()