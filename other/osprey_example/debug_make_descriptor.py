from aglaia import wrappers
import numpy as np
from sklearn.base import clone
import glob

estimator = wrappers.OAMNN(iterations=10, batch_size=100, scoring_function="mae",
                        hidden_layer_sizes=[20, 10], learning_rate=0.003441414420250073, representation="slatm")

filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:10]
filenames.sort()
energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:10]
estimator.generate_compounds(filenames)
estimator.set_properties(energies)

idx = np.arange(0,10)

print("\n The type of the descriptor before fitting is:")
print(type(estimator.descriptor))
print("\n The type of zs before fitting is:")
print(type(estimator.zs))

# estimator.fit(idx)
estimator.generate_descriptor()

print("\n The type of the descriptor after generating descriptor is:")
print(type(estimator.descriptor))
print("\n The type of zs after generating descriptor is:")
print(type(estimator.zs))

estimator.get_params()

dolly = clone(estimator)


print("\n The type of the descriptor after cloning is:")
print(type(estimator.descriptor))
print("\n The type of zs after cloning is:")
print(type(estimator.zs))