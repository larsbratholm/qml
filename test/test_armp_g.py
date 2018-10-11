# MIT License
#
# Copyright (c) 2018 Silvia Amabilino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This test checks if all the ways of setting up the estimator ARMP work.
"""

import numpy as np
from qml.aglaia.aglaia import ARMP_G
from qml.utils.utils import InputError
import glob
from qml.utils.utils import is_array_like
import os
import shutil
import tensorflow as tf

def test_set_representation():
    """
    This function tests the function _set_representation.
    """
    try:
        ARMP_G(representation_name='acsf', representation_params={'slatm_sigma12': 0.05})
        raise Exception
    except InputError:
        pass

    try:
        ARMP_G(representation_name='coulomb_matrix')
        raise Exception
    except InputError:
        pass

    try:
        ARMP_G(representation_name='slatm')
        raise Exception
    except InputError:
        pass

    parameters = {'rcut': 10.0, 'acut': 10.0, 'nRs2': 3, 'nRs3': 3, 'nTs': 2,
                                      'zeta': 3.0, 'eta': 2.0}

    estimator = ARMP_G(representation_name='acsf', representation_params=parameters)

    assert estimator.representation_name == 'acsf'

    for key, value in estimator.acsf_parameters.items():
        if is_array_like(value):
            assert np.all(estimator.acsf_parameters[key] == parameters[key])
        else:
            assert estimator.acsf_parameters[key] == parameters[key]

def test_set_properties():
    """
    This test checks that the set_properties function sets the correct properties.
    :return:
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    energies = np.loadtxt(test_dir + '/CN_isopentane/prop_kjmol_training.txt',
                          usecols=[1])

    estimator = ARMP_G()

    assert estimator.properties == None

    estimator.set_properties(energies)

    assert np.all(estimator.properties == energies)

def test_fit():
    """
    This function tests the first way of fitting the representation: the data is passed by first creating compounds and then
    the representations are created from the compounds.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    filenames = glob.glob(test_dir + "/CN_isopentane/*.xyz")
    energies = np.loadtxt(test_dir + '/CN_isopentane/prop_kjmol_training.txt',
                          usecols=[1])
    data = np.load(test_dir + "/data/CN_isopentane_forces.npz")
    filenames.sort()
    forces =  data["arr_3"][:5]

    estimator = ARMP_G(representation_name="acsf", iterations=2)
    estimator.generate_compounds(filenames[:5])
    estimator.set_properties(energies[:5])
    estimator.set_gradients(forces)

    idx = list(range(5))
    estimator.fit(idx)

def test_score():
    """
    This function tests that all the scoring functions work.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    filenames = glob.glob(test_dir + "/CN_isopentane/*.xyz")
    energies = np.loadtxt(test_dir + '/CN_isopentane/prop_kjmol_training.txt',
                          usecols=[1])
    data = np.load(test_dir + "/data/CN_isopentane_forces.npz")
    filenames.sort()
    forces =  data["arr_3"][:5]
    idx = list(range(5))

    estimator_1 = ARMP_G(scoring_function='mae', iterations=2)
    estimator_1.generate_compounds(filenames[:5])
    estimator_1.set_properties(energies[:5])
    estimator_1.set_gradients(forces)
    estimator_1.fit(idx)
    estimator_1.score(idx)

    estimator_2 = ARMP_G(scoring_function='r2', iterations=2)
    estimator_2.generate_compounds(filenames[:5])
    estimator_2.set_properties(energies[:5])
    estimator_2.set_gradients(forces)
    estimator_2.fit(idx)
    estimator_2.score(idx)

    estimator_3 = ARMP_G(scoring_function='rmse', iterations=2)
    estimator_3.generate_compounds(filenames[:5])
    estimator_3.set_properties(energies[:5])
    estimator_3.set_gradients(forces)
    estimator_3.fit(idx)
    estimator_3.score(idx)

def test_predict():
    test_dir = os.path.dirname(os.path.realpath(__file__))

    filenames = glob.glob(test_dir + "/CN_isopentane/*.xyz")
    energies = np.loadtxt(test_dir + '/CN_isopentane/prop_kjmol_training.txt',
                          usecols=[1])[:5]
    data = np.load(test_dir + "/data/CN_isopentane_forces.npz")
    filenames.sort()
    filenames = filenames[:5]
    forces =  data["arr_3"][:5]
    idx = list(range(5))

    estimator = ARMP_G(representation_name="acsf", iterations=2)
    estimator.generate_compounds(filenames)
    estimator.set_properties(energies)
    estimator.set_gradients(forces)
    estimator.fit(idx)
    energies_pred, dy_pred = estimator.predict(idx)

    assert energies.shape == energies_pred.shape
    assert forces.shape == dy_pred.shape

def test_retraining():

    tf.reset_default_graph()

    xyz = np.array([[[0, 1, 0], [0, 1, 1], [1, 0, 1]],
                    [[1, 2, 2], [3, 1, 2], [1, 3, 4]],
                    [[4, 1, 2], [0.5, 5, 6], [-1, 2, 3]]])
    zs = np.array([[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]])

    ene_true = np.array([0.5, 0.9, 1.0])

    forces = np.array([[[0, 1, 0], [0, 1, 1], [1, 0, 1]],
                    [[1, 2, 2], [3, 1, 2], [1, 3, 4]],
                    [[4, 1, 2], [0.5, 5, 6], [-1, 2, 3]]])

    acsf_param = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
    estimator = ARMP_G(iterations=2, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation_name='acsf',
                     representation_params=acsf_param)

    estimator.set_xyz(xyz)
    estimator.set_gradients(forces)
    estimator.set_classes(zs)
    estimator.set_properties(ene_true)

    idx = list(range(xyz.shape[0]))

    estimator.fit(idx)
    estimator.save_nn(save_dir="temp")

    pred_ene_1, pred_f_1 = estimator.predict(idx)

    estimator.loaded_model = True

    estimator.fit(idx)

    pred_ene_2, pred_f_2 = estimator.predict(idx)

    new_estimator = ARMP_G(iterations=2, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation_name='acsf',
                         representation_params=acsf_param)

    new_estimator.set_xyz(xyz)
    new_estimator.set_classes(zs)
    new_estimator.set_properties(ene_true)
    new_estimator.set_gradients(forces)

    new_estimator.load_nn("temp")

    pred_ene_3, pred_f_3 = new_estimator.predict(idx)

    new_estimator.fit(idx)

    pred_ene_4, pred_f_4 = new_estimator.predict(idx)

    shutil.rmtree("temp")

    assert np.all(np.isclose(pred_ene_1, pred_ene_3, rtol=1.e-6))
    assert np.all(np.isclose(pred_f_1, pred_f_3, rtol=1.e-6))
    assert np.all(np.isclose(pred_ene_2, pred_ene_4, rtol=1.e-6))
    assert np.all(np.isclose(pred_f_2, pred_f_4, rtol=1.e-6))


if __name__ == "__main__":
    test_set_representation()
    test_set_properties()
    test_fit()
    test_score()
    test_predict()
    test_retraining()