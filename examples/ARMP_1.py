"""
This script shows how to set up the ARMP estimator where the XYZ data is used to make QML compounds and the local
descriptors are generated from the QML compounds and then stored.
"""

import aglaia
import glob
import numpy as np

## ------------- ** Loading the data ** ---------------

filenames = glob.glob("/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/*.xyz")
energies = np.loadtxt('/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/prop_kjmol_training.txt', usecols=[1])
filenames.sort()

## ------------- ** Setting up the estimator ** ---------------

estimator = aglaia.ARMP(iterations=1000, representation='acsf', descriptor_params={"rad_rs": np.arange(0,10, 0.1), "ang_rs": np.arange(0.5, 10.5, 0.1),
"theta_s": np.arange(0, 5, 0.1)}, tensorboard=True, tensorboard_subdir="/Users/walfits/Desktop/tensorboard/")

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