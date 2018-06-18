"""
This script shows how to set up the MRMP estimator where the descriptor is set directly and stored in the class.
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

estimator.set_descriptors(descriptors=descriptor)
estimator.set_properties(energies)


