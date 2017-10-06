import unittest
import inverse_dist as des
import tensorflow as tf
import numpy as np

class InvTest(unittest.TestCase):

    def test_inv(self):
        """
        This function tests the inverse matrix function
        """

        xyz_test = tf.constant([[0., 0., 0., 1., 0., 0.], [0., 0., 0., 2., 0., 0.]])
        exp_result = [[1.0], [2.0]]
        mat = des.inverse_dist(xyz_test, n_samples=2, n_atoms=2)

        sess = tf.Session()
        actual_result = sess.run(mat)

        self.assertTrue(np.all(np.isclose(exp_result, actual_result)))





if __name__ == "__main__":
    unittest.main()