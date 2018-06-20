"""
This script shows how to set up the MRMP estimator where the data to be fitted is passed directly to the fit function.
"""

import aglaia
import numpy as np

## ------------- ** Loading the data ** ---------------

# The data loaded contains 100 samples of the CN + isobutane data set in unsorted CM representation
data = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopent_light_UCM.npz")

descriptor = data["arr_0"]
energies = data["arr_1"]

## ------------- ** Setting up the estimator ** ---------------

estimator = aglaia.MRMP()

##  ------------- ** Fitting to the data ** ---------------

estimator.fit(descriptor, energies)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(descriptor, energies)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(descriptor)

## ------------- ** Correlation plot ** ---------------

import matplotlib.pyplot as plt

plt.scatter(energies, energies_predict)
plt.show()