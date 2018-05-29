from aglaia import wrappers
import numpy as np
import glob
from sklearn import model_selection as modsel
from sklearn.base import clone
import time

rad_rs = np.arange(0,10, 5)
ang_rs = np.arange(0.5, 10.5, 5)
theta_s = np.arange(0, 5, 5)
# rad_rs = [0.0]
# ang_rs = [0.0]
# theta_s = [3.0]

estimator = wrappers.OAMNN(iterations=10, batch_size=70, scoring_function="mae", tensorboard=True, l2_reg=0.03906198710720926,
                        hidden_layer_sizes=[50, 20], learning_rate=0.005, representation="acsf", slatm_dgrid1 = 0.12,
                           radial_rs=rad_rs, angular_rs=ang_rs, theta_s=theta_s, zeta=8.0, eta=4.0)

# filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:100]
# filenames = glob.glob("/Volumes/Transcend/data_sets/vr_ccsd/*.xyz")
filenames = glob.glob("/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/*.xyz")
filenames.sort()
# energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:100]
# energies = np.loadtxt('/Volumes/Transcend/data_sets/vr_ccsd/properties.txt', usecols=[1])
energies = np.loadtxt('/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/prop_kjmol_training.txt', usecols=[1])

estimator.generate_compounds(filenames)
estimator.set_properties(energies)

start_time = time.time()
estimator.generate_descriptor(descriptor_batch_size=20)
end_time = time.time()

print("Generating the descriptor took %s." % (end_time-start_time) )

idx = np.arange(0,len(filenames))

idx_train, idx_test = modsel.train_test_split(idx, test_size=0.3)

estimator.fit(idx_train)
estimator.score(idx_test)
y_pred = estimator.predict(idx_train)
estimator.correlation_plot(np.squeeze(y_pred), energies[idx_train])

# estimator.plot_cost()
#
# # y_pred = estimator.predict(idx_test)
#
# # estimator.correlation_plot(np.squeeze(y_pred), energies[idx_test])
#
#
# print(train_scores)
