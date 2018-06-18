"""
This script shows how to set up the MRMP estimator where the XYZ data is used to make QML compounds and global descriptors
are generated from the QML compounds and stored.
"""


import aglaia
import glob
import numpy as np

## ------------- ** Loading the data ** ---------------

filenames = glob.glob("/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/*.xyz")
energies = np.loadtxt('/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/prop_kjmol_training.txt', usecols=[1])
filenames.sort()

## ------------- ** Setting up the estimator ** ---------------

estimator = aglaia.MRMP(representation='unsorted_coulomb_matrix')

estimator.generate_compounds(filenames[:100])
estimator.set_properties(energies[:100])

estimator.generate_descriptors()