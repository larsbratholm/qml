"""This example shows how to use the OMNN estimator from the aglaia wrapper. It shows how to create the estimator,
how to train it, how to score the performance of it and how to use it to make predictions.
It uses a tiny data set with 100 molecules."""

from aglaia import wrappers
from sklearn import model_selection as modsel
import glob
import numpy as np

## ------------- ** Loading the data ** ---------------

filenames = glob.glob("/Volumes/Transcend/data_sets/vr_ccsd/*.xyz")[:100]

energies = np.loadtxt('/Volumes/Transcend/data_sets/vr_ccsd/properties.txt', usecols=[1])[:100]

## ------------- ** Generating the estimator ** ---------------

estimator = wrappers.OMNN(iterations=400, batch_size=50, scoring_function="mae", l2_reg=0.03906198710720926,
                        hidden_layer_sizes=[50, 20], learning_rate=0.005, representation="unsorted_coulomb_matrix")

estimator.generate_compounds(filenames)
estimator.set_properties(energies)

## ------------- ** Generating the descriptor ** ---------------

estimator.generate_descriptor()

## ------------- ** Fitting the estimator ** ---------------

idx = np.arange(len(filenames))
idx_train, idx_test = modsel.train_test_split(idx, test_size=0.3)

estimator.fit(idx_train)

## ------------ ** Scoring the estimator ** --------------

score = estimator.score(idx_train)

print("The score is: %s" % (score))

## ------------ ** Using the estimator to predict properties ** ------------

predicted_energies = estimator.predict(idx_test)
