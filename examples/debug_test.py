import pickle
import glob
import numpy as np
from sklearn.base import clone
from aglaia import wrappers
import tensorflow as tf

estimator = wrappers.OSPMRMP(batch_size = 100, representation = "unsorted_coulomb_matrix",
                    hl1=12, optimiser=tf.train.AdadeltaOptimizer, scoring_function='mae',
                    iterations=6000, slatm_sigma1 = 0.1, activation_function=tf.nn.relu)

# Printing the number of iterations
print("\n The number of iterations after loading is")
print(estimator.iterations)
print("\n The slatm after loading is")
print(estimator.slatm_sigma1)
print("\n The optimiser before cloning is:")
print(estimator.optimiser)
print("\n The activation function before cloning is:")
print(estimator.activation_function)

estimator.get_params()

dolly = clone(estimator)
print("\n The slatm after cloning is:")
print(dolly.slatm_sigma1)
print("\n The iterations after cloning are:")
print(dolly.iterations)
print("\n The optimiser after cloning is:")
print(dolly.optimiser)
print("\n The activation function after cloning is:")
print(dolly.activation_function)

