"""
This test checks if all the ways of setting up the estimator MRMP work.
"""


import numpy as np
import aglaia
import joblib
from aglaia import InputError
import glob

def test_set_representation():
    """
    This function tests the function _set_representation.
    """
    try:
        aglaia.MRMP(representation='unsorted_coulomb_matrix', descriptor_params={'slatm_sigma1': 0.05})
        raise Exception
    except InputError:
        pass

    try:
        aglaia.MRMP(representation='coulomb_matrix')
        raise Exception
    except InputError:
        pass

    try:
        aglaia.MRMP(representation='slatm', descriptor_params={'slatm_alchemy': 0.05})
        raise Exception
    except InputError:
        pass

    parameters ={'slatm_sigma1': 0.07, 'slatm_sigma2': 0.04, 'slatm_dgrid1': 0.02, 'slatm_dgrid2': 0.06,
                                'slatm_rcut': 5.0, 'slatm_rpower': 7, 'slatm_alchemy': True}

    estimator = aglaia.MRMP(representation='slatm', descriptor_params=parameters)

    assert estimator.representation == 'slatm'
    assert estimator.slatm_parameters == parameters

def test_set_properties():
    """
    This test checks that the set_properties function sets the correct properties.
    :return:
    """

    energies = np.loadtxt('/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/prop_kjmol_training.txt',
                          usecols=[1])

    estimator = aglaia.MRMP(representation='unsorted_coulomb_matrix')

    assert estimator.properties == None

    estimator.set_properties(energies)

    assert np.all(estimator.properties == energies)

def test_set_descriptor():
    """
    This test checks that the set_descriptor function works as expected.
    :return:
    """

    data_correct = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopent_light_UCM.npz")
    data_incorrect = joblib.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_ch4cn_light.bz")
    descriptor_correct = data_correct["arr_0"]
    descriptor_incorrect = data_incorrect["descriptor"]


    estimator = aglaia.MRMP()

    assert estimator.descriptor == None

    estimator.set_descriptors(descriptors=descriptor_correct)

    assert np.all(estimator.descriptor == descriptor_correct)

    # Pass a descriptor with the wrong shape
    try:
        estimator.set_descriptors(descriptors=descriptor_incorrect)
        raise Exception
    except InputError:
        pass

def test_fit_1():
    """
    This function tests the first way of fitting the descriptor: the data is passed by first creating compounds and then
    the descriptors are created from the compounds.
    """

    filenames = glob.glob("/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/*.xyz")
    energies = np.loadtxt('/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/prop_kjmol_training.txt',
                          usecols=[1])
    filenames.sort()

    estimator = aglaia.MRMP()
    estimator.generate_compounds(filenames[:100])
    estimator.set_properties(energies[:100])
    estimator.generate_descriptors()

    idx = np.arange(0, 100)
    estimator.fit(idx)

def test_fit_2():
    """
    This function tests the second way of fitting the descriptor: the data is passed by storing the compounds in the
    class.
    """
    data = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator = aglaia.MRMP()
    estimator.set_descriptors(descriptors=descriptor)
    estimator.set_properties(energies)

    idx = np.arange(0, 100)
    estimator.fit(idx)

def test_fit_3():
    """
    This function tests the thrid way of fitting the descriptor: the data is passed directly to the fit function.
    """
    data = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator = aglaia.MRMP()
    estimator.fit(descriptor, energies)

def test_score_3():
    """
    This function tests that all the scoring functions work.
    """

    data = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator_1 = aglaia.MRMP(scoring_function='mae')
    estimator_1.fit(descriptor, energies)
    estimator_1.score(descriptor, energies)

    estimator_2 = aglaia.MRMP(scoring_function='r2')
    estimator_2.fit(descriptor, energies)
    estimator_2.score(descriptor, energies)

    estimator_3 = aglaia.MRMP(scoring_function='rmse')
    estimator_3.fit(descriptor, energies)
    estimator_3.score(descriptor, energies)

def test_predict_3():
    data = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopent_light_UCM.npz")
    descriptor = data["arr_0"]
    energies = data["arr_1"]

    estimator = aglaia.MRMP()
    estimator.fit(descriptor, energies)
    energies_pred = estimator.predict(descriptor)

    assert energies.shape == energies_pred.shape


if __name__ == "__main__":

    test_set_properties()
    test_set_descriptor()
    test_set_representation()
    test_fit_1()
    test_fit_2()
    test_fit_3()
    test_score_3()
    test_predict_3()