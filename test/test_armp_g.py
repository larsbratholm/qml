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

def test_set_representation():
    """
    This test checks that the set_representation function works as expected.
    :return:
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data_incorrect = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    data_correct = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    representation_correct = np.asarray(data_correct["arr_0"])
    representation_incorrect = np.asarray(data_incorrect["arr_0"])

    dgdr_correct = np.ones((3, 4, 5, 4, 3))
    dgdr_incorrect = np.ones((3, 1, 2, 3, 4))


    estimator = ARMP_G()

    assert estimator.g == None

    estimator._set_representation(g=representation_correct)

    assert np.all(estimator.g == representation_correct)

    # Pass a representation with the wrong shape
    try:
        estimator._set_representation(g=representation_incorrect)
        raise Exception
    except InputError:
        pass

def test_fit_1():
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
    forces =  data["arr_3"][:2]

    estimator = ARMP_G(representation_name="acsf")
    estimator.generate_compounds(filenames[:2])
    estimator.set_properties(energies[:2])
    estimator.set_gradients(forces)
    estimator.generate_representation()

    idx = np.arange(0, 2)
    estimator.fit(idx)

def test_fit_2():
    """
    This function tests the second way of fitting the representation: the data is passed by storing the compounds in the
    class.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_acsf_light.npz")
    representation = data["arr_0"]
    dg_dr = data["arr_1"]
    energies = data["arr_2"]
    forces = data["arr_3"]
    classes = data["arr_4"]

    estimator = ARMP_G()
    estimator._set_representation(g=representation)
    estimator.set_classes(classes=classes)
    estimator.set_properties(energies)
    estimator.set_gradients(forces)

    idx = np.arange(0, 2)
    estimator.fit(idx)

def test_fit_3():
    """
    This function tests the thrid way of fitting the representation: the data is passed directly to the fit function.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_acsf_light.npz")
    representation = data["arr_0"]
    dg_dr = data["arr_1"]
    energies = data["arr_2"]
    forces = data["arr_3"]
    classes = data["arr_4"]

    estimator = ARMP_G()
    estimator.fit(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)

def test_score_3():
    """
    This function tests that all the scoring functions work.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_acsf_light.npz")
    representation = data["arr_0"]
    dg_dr = data["arr_1"]
    energies = data["arr_2"]
    forces = data["arr_3"]
    classes = data["arr_4"]

    estimator_1 = ARMP_G(scoring_function='mae')
    estimator_1.fit(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)
    estimator_1.score(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)

    estimator_2 = ARMP_G(scoring_function='r2')
    estimator_2.fit(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)
    estimator_2.score(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)

    estimator_3 = ARMP_G(scoring_function='rmse')
    estimator_3.fit(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)
    estimator_3.score(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)

def test_predict_3():
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data = np.load(test_dir + "/data/local_acsf_light.npz")
    representation = data["arr_0"]
    dg_dr = data["arr_1"]
    energies = data["arr_2"]
    forces = data["arr_3"]
    classes = data["arr_4"]

    estimator = ARMP_G()
    estimator.fit(x=representation, y=energies, classes=classes, dy=forces, dgdr=dg_dr)
    energies_pred, dy_pred = estimator.predict(x=representation, classes=classes, dgdr=dg_dr)

    assert energies.shape == energies_pred.shape
    assert forces.shape == dy_pred.shape

if __name__ == "__main__":
    test_set_representation()
    test_set_properties()
    test_set_representation()
    test_fit_1()
    test_fit_2()
    test_fit_3()
    test_score_3()
    test_predict_3()