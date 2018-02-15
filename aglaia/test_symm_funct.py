"""
This file contains tests for the atom centred symmetry function module.
"""

import tensorflow as tf
import numpy as np

import symm_funct

## ---------------------- ** Functions needed by the test functions ** ------------------------

def distance(r1, r2):
    diff = r2-r1
    return np.linalg.norm(diff)

def fc(r_ij, r_c):
    if r_ij < r_c:
        f_c = 0.5 * (np.cos(np.pi * r_ij / r_c) + 1.0)
    else:
        f_c = 0.0
    return f_c

def get_theta(xyz_i, xyz_j, xyz_k):
    r_ij = xyz_j - xyz_i
    r_ik = xyz_k - xyz_i
    numerator = np.dot(r_ij, r_ik)
    denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_ik)
    costheta = numerator/denominator
    theta = np.arccos(costheta)
    return theta

def numpy_salpha_ang(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    """
    This does the angular part of the symmetry function as mentioned here: https://arxiv.org/pdf/1711.06385.pdf
    It implements in in numpy very inefficiently.

    :param xyzs: np.array of shape (n_samples, n_atoms, 3) with the coordinates of the atoms
    :param Zs: np.array of shape (n_samples, n_atoms) with the atomic umbers of the atoms
    :param angular_cutoff: scalar with the cut-off of the fc term
    :param angular_rs: np.array of shape (n_rs, )
    :param theta_s: np.array of shape (n_thetas,)
    :param zeta: scalar
    :param eta: scalar
    :return: np.array of shape (n_samples, n_atoms, n_rs*n_thetas)
    """
    # Useful numbers
    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]

    total_descriptor = []
    for sample in range(n_samples):
        sample_descriptor = []
        for i in range(n_atoms):  # Loop over main atom
            atom_descriptor = []
            for angular_rs_value in angular_rs:
                for theta_s_value in theta_s:
                    g_sum = 0
                    for j in range(n_atoms):  # Loop over 1st neighbour
                        if j == i:
                            continue
                        for k in range(n_atoms):  # Loop over 2nd neighbour
                            if k == j or k == i:
                                continue

                            r_ij = distance(xyzs[sample, i, :], xyzs[sample, j, :])
                            r_ik = distance(xyzs[sample, i, :], xyzs[sample, k, :])

                            theta_ijk = get_theta(xyzs[sample, i, :], xyzs[sample, j, :], xyzs[sample, k, :])

                            term1 = np.power((1.0 + np.cos(theta_ijk - theta_s_value)), zeta)
                            exponent = - eta * np.power(0.5 * (r_ij + r_ik) - angular_rs_value, 2.0)
                            term2 = np.exp(exponent)
                            term3 = fc(r_ij, angular_cutoff) * fc(r_ik, angular_cutoff)

                            g_term = term1 * term2 * term3
                            g_sum += g_term

                    atom_descriptor.append(g_sum * np.power(2.0, 1.0 - zeta))
            sample_descriptor.append(atom_descriptor)
        total_descriptor.append(sample_descriptor)

    total_descriptor = np.asarray(total_descriptor)
    return total_descriptor

## ---------------------- ** Test functions ** ------------------------

def test_g2():
    """
    This test uses a system of 4 Argon atoms placed at positions that make it easier to see if the descriptor is being
    constructed correctly.

    :return:
    """
    # Input data
    xyzs = [[[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]]

    Zs = [[18, 18, 18, 18]]

    # Parameters for the ACSF
    eta = tf.constant(1.0, dtype=tf.float32)
    rs = tf.constant(0.0, dtype=tf.float32)
    radial_cutoff = tf.constant(500.0, dtype=tf.float32)

    # Turning the data into tensorflow
    xyzs_tf = tf.constant(xyzs, dtype=tf.float32)  # (n_samples, n_atoms, 3)
    Zs_tf = tf.constant(Zs, dtype=tf.int32)

    # Making the descriptor
    g2_tf = symm_funct.symm_func_g2(xyzs_tf, Zs_tf, radial_cutoff, rs, eta)

    expected_g2 = [[0.36787584],
                     [0.36787584],
                     [0.36787584],
                     [0.36787584],
                     [0.13533266],
                     [0.13533266],
                     [0.36787584],
                     [0.13533266],
                     [0.13533266],
                     [0.36787584],
                     [0.13533266],
                     [0.13533266]]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    g2 = sess.run(g2_tf)

    np.testing.assert_array_almost_equal(g2, expected_g2, decimal=5)

def test_salpha_ang():
    """
    This test checks that the radial part of the symmetry function descriptor from the tensorflow implementation agrees
    with the inefficient numpy implementation.
    """

    xyzs_list = [[[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]]

    Zs_list = [[18, 18, 18, 18]]

    # Data for numpy
    xyzs_np = np.asarray(xyzs_list)
    Zs_np = np.asarray(Zs_list)

    eta = 2.0
    zeta = 3.0
    theta_s = np.array([3.0, 2.0])
    angular_rs = np.array([0.0, 0.1, 0.2])
    angular_cutoff = 500.0

    # Data for tensorflow
    xyzs_tf = tf.constant(xyzs_list, dtype=tf.float32)
    Zs_tf = tf.constant(Zs_list, dtype=tf.float32)
    eta_tf = tf.constant(eta, dtype=tf.float32)
    zeta_tf = tf.constant(zeta, dtype=tf.float32)
    thetas_tf = tf.constant(theta_s, dtype=tf.float32)
    angular_rs_tf = tf.constant(angular_rs, dtype=tf.float32)
    angular_cutoff_tf = tf.constant(angular_cutoff, dtype=tf.float32)

    # Constructing descriptors
    np_descriptor = numpy_salpha_ang(xyzs_np, Zs_np, angular_cutoff, angular_rs, theta_s, zeta, eta)
    tf_descriptor = symm_funct.symm_func_salpha_ang(xyzs_tf, Zs_tf, angular_cutoff_tf, angular_rs_tf, thetas_tf, zeta_tf, eta_tf)

    # Running the tensorflow descriptor
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    final_tf_descriptor = sess.run(tf_descriptor)

    np.testing.assert_array_almost_equal(np_descriptor, final_tf_descriptor, decimal=6)



if __name__ == "__main__":
    test_salpha_ang()