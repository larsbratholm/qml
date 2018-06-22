"""
This script shows how to set up the ARMP estimator where the data to be fitted is passed directly to the fit function.
"""

import aglaia
import joblib

## ------------- ** Loading the data ** ---------------

data = joblib.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_ch4cn_light.bz")

descriptor = data["descriptor"]
energies = data["energies"]
zs = data["zs"]

## ------------- ** Setting up the estimator ** ---------------

estimator = aglaia.ARMP(iterations=15000, l2_reg=0.0, learning_rate=0.005, hidden_layer_sizes=(40, 20, 10))

##  ------------- ** Fitting to the data ** ---------------

estimator.fit(x=descriptor, y=energies, classes=zs)

##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(x=descriptor, y=energies, classes=zs)

print("The mean absolute error is %s kJ/mol." % (str(-score)))

energies_predict = estimator.predict(x=descriptor, classes=zs)

## ------------- ** Correlation plot ** ---------------

import matplotlib.pyplot as plt

plt.scatter(energies, energies_predict)
plt.show()