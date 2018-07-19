"""
This script shows how to set up the ARMP_G estimator where the xyz, nuclear charges, energies and forces are all set in
advance and are then used for fitting using the indices.
"""

from qml.aglaia.aglaia import ARMP_G
import glob
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir+"/../test/data/CN_isopentane_forces.npz")

xyz = data["arr_0"]
zs = data["arr_1"]
ene = data["arr_2"]
forces = data["arr_3"]

print(xyz.shape, zs.shape, ene.shape, forces.shape)

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP_G(iterations=1, representation='acsf', descriptor_params={"radial_rs": np.arange(0,10, 2), "angular_rs": np.arange(0.5, 10.5, 2),
"theta_s": np.arange(0, 3.14, 2.5)}, tensorboard=False, store_frequency=1, batch_size=10)

estimator.set_xyz(xyz)
estimator.set_classes(zs)
estimator.set_properties(ene)
estimator.set_gradients(forces)

##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,5)

estimator.fit(idx)

# ##  ------------- ** Predicting and scoring ** ---------------

score = estimator.score(idx)

print("The mean absolute error is %s (kcal/mol)." % (str(-score)))

energies_predict = estimator.predict(idx)
