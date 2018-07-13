"""
This script shows how to set up the ARMP estimator where the XYZ data is used to make QML compounds and the local
descriptors are generated from the QML compounds and then stored.
"""
from qml.aglaia.aglaia import ARMP_G
import glob
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

data = np.load("/Volumes/Transcend/repositories/Aglaia/data/CN_isopentane_forces.npz")

xyz = data["arr_0"]
zs = data["arr_1"]
ene = data["arr_2"]
forces = data["arr_3"]

print(xyz.shape, zs.shape, ene.shape, forces.shape)

## ------------- ** Setting up the estimator ** ---------------

estimator = ARMP_G(iterations=100, representation='acsf', descriptor_params={"radial_rs": np.arange(0,10, 0.5), "angular_rs": np.arange(0.5, 10.5, 0.5),
"theta_s": np.arange(0, 3.14, 0.2)}, tensorboard=True, batch_size=10)

estimator.set_xyz(xyz)
estimator.set_classes(zs)
estimator.set_properties(ene)
estimator.set_gradients(forces)




##  ------------- ** Fitting to the data ** ---------------

idx = np.arange(0,100)

estimator.fit(idx)


# ##  ------------- ** Predicting and scoring ** ---------------
#
# score = estimator.score(idx)
#
# print("The mean absolute error is %s kJ/mol." % (str(-score)))
#
# energies_predict = estimator.predict(idx)