import unittest
import inverse_dist as des
import tensorflow as tf
import numpy as np

class Test(unittest.TestCase):

    def test_dist_mat(self):
        """
        This function tests the inverse matrix function
        """

        xyz_test = tf.constant([[0., 0., 0., 1., 0., 0.], [0., 0., 0., 2., 0., 0.]])
        exp_result = [[1.0], [2.0]]
        mat = des.dist_mat(xyz_test, n_samples=2, n_atoms=2)

        sess = tf.Session()
        actual_result = sess.run(mat)

        self.assertTrue(np.all(np.isclose(exp_result, actual_result)))


    def test_inv_mat(self):
        xyz_test = tf.constant([[0., 0., 0., 1., 0., 0.], [0., 0., 0., 2., 0., 0.]])
        exp_result = [[1.0], [0.5]]

        mat = des.inv_dist(xyz_test, n_samples=2, n_atoms=2)

        sess = tf.Session()
        actual_result = sess.run(mat)

        self.assertTrue(np.all(np.isclose(exp_result, actual_result)))


if __name__ == "__main__":
    unittest.main()