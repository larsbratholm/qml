import pickle
import glob
import numpy as np
from sklearn.base import clone
from aglaia import wrappers
import tensorflow as tf

estimator = wrappers.OSPMRMP(batch_size = 100, representation = "unsorted_coulomb_matrix",
                    hl1=12, optimiser=tf.train.AdadeltaOptimizer, scoring_function='mae',
                    iterations=6000, slatm_sigma1 = 0.1)

# Printing the number of iterations
print("\n The number of iterations after loading is")
print(estimator.iterations)
print("\n The slatm after loading is")
print(estimator.slatm_sigma1)

estimator.get_params()

dolly = clone(estimator)
# print("\n The slatm after cloning is:")
# print(dolly.slatm_sigma1)
# print("\n The iterations after cloning are:")
# print(dolly.iterations)

# klass = estimator.__class__
# new_obj_params = estimator.get_params(deep=False)
# new_obj = klass(**new_obj_params)
# print(new_obj)

