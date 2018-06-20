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

estimator = aglaia.MRMP(representation='slatm', descriptor_params={'slatm_dgrid2': 0.06, 'slatm_dgrid1': 0.06})

estimator.generate_compounds(filenames[:100])
estimator.set_properties(energies[:100])

estimator.generate_descriptors()

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(idx)

## ------------- ** Correlation plot ** ---------------

import matplotlib.pyplot as plt

plt.scatter(energies[:100], energies_predict)
plt.show()