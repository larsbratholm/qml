"""
This file contains tests for the atom centred symmetry function module.
"""

import tensorflow as tf
import numpy as np

from aglaia import symm_funct
from aglaia.tests import np_symm_funct
from aglaia.tests import tensormol_symm_funct


def test_acsf_1():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and numpy.
    The test system consists of 5 configurations of CH4 + CN radical.

    :return: None
    """

    input_data = "data_test_acsf.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyzs, zs, elements, element_pairs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t)

    acsf_np = np_symm_funct.generate_acsf(xyzs, zs, elements, element_pairs)

    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]

    for i in range(n_samples):
        for j in range(n_atoms):
            acsf_np_sort = np.sort(acsf_np[i][j])
            acsf_tf_sort = np.sort(acsf_tf[i][j])
            np.testing.assert_array_almost_equal(acsf_np_sort, acsf_tf_sort, decimal=4)


def test_acsf_2():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and tensormol.
    The test system consists of 4 atoms at very predictable positions.

    :return:
    """

    input_data = "data_test_acsf_01.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyzs, zs, elements, element_pairs)
    acsf_tm_t = tensormol_symm_funct.tensormol_acsf(xyzs, zs, elements, element_pairs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t)
    acsf_tm = sess.run(acsf_tm_t)

    acsf_tf = np.reshape(acsf_tf, acsf_tm.shape)

    for i in range(acsf_tm.shape[0]):
        acsf_tm_sort = np.sort(acsf_tm[i])
        acsf_tf_sort = np.sort(acsf_tf[i])
        np.testing.assert_array_almost_equal(acsf_tm_sort, acsf_tf_sort, decimal=1)


