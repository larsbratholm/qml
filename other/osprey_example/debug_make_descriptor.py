from aglaia import wrappers
import numpy as np
from sklearn.base import clone
import glob

rs = [0.0, 0.4818181818181818, 0.9636363636363636, 1.4454545454545453, 1.9272727272727272, 2.409090909090909, 2.8909090909090907, 3.372727272727273, 3.8545454545454545, 4.336363636363636, 4.818181818181818, 5.3, 5.781818181818181, 6.263636363636363, 6.745454545454546, 7.2272727272727275, 7.709090909090909, 8.19090909090909, 8.672727272727272, 9.154545454545454, 9.636363636363637, 10.118181818181817]

estimator = wrappers.OAMNN(iterations=30000, batch_size=100, scoring_function="mae",
                        hidden_layer_sizes=[50, 20], learning_rate=0.005, representation="acsf", radial_cutoff=7.0, angular_cutoff=7.0,
                          radial_rs=rs, angular_rs=rs, theta_s=[0.0, 0.785, 1.57, 2.355, 3.14, 3.925, 4.71, 5.495], zeta=8.0, eta=4.0)

# filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:100]
filenames = glob.glob("/Volumes/Transcend/data_sets/vr_ccsd/*.xyz")[:100]
filenames.sort()
# energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:100]
energies = np.loadtxt('/Volumes/Transcend/data_sets/vr_ccsd/properties.txt', usecols=[1])[:100]
estimator.generate_compounds(filenames)
estimator.set_properties(energies)

idx = np.arange(0,100)

estimator.generate_descriptor()
estimator.fit(idx)
estimator.plot_cost()

y_pred = estimator.predict(idx)

estimator.correlation_plot(np.squeeze(y_pred), energies)

# print("\n The type of the descriptor before fitting is:")
# print(type(estimator.descriptor))
# print("\n The type of zs before fitting is:")
# print(type(estimator.zs))

# estimator.fit(idx)
# estimator.generate_descriptor()
#
# print("\n The type of the descriptor after generating descriptor is:")
# print(type(estimator.descriptor))
# print("\n The type of zs after generating descriptor is:")
# print(type(estimator.zs))
#
# estimator.get_params()
#
# dolly = clone(estimator)
#
#
# print("\n The type of the descriptor after cloning is:")
# print(type(estimator.descriptor))
# print("\n The type of zs after cloning is:")
# print(type(estimator.zs))