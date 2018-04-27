from aglaia import wrappers
import numpy as np
import glob
from sklearn import model_selection as modsel
from sklearn.base import clone
import time

rad_rs = np.arange(0,10, 0.6)
ang_rs = np.arange(0.5, 10.5, 0.6)
theta_s = np.arange(0, 5, 0.6)
# rad_rs = [0.0]
# ang_rs = [0.0]
# theta_s = [3.0]

estimator = wrappers.OAMNN(iterations=600, batch_size=70, scoring_function="mae", tensorboard=True, l2_reg=0.03906198710720926,
                        hidden_layer_sizes=[50, 20], learning_rate=0.005, representation="acsf", slatm_dgrid1 = 0.12,
                           radial_rs=rad_rs, angular_rs=ang_rs, theta_s=theta_s, zeta=8.0, eta=4.0)

# filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:100]
filenames = glob.glob("/Volumes/Transcend/data_sets/vr_ccsd/*.xyz")[:100]
# filenames = glob.glob("/Volumes/Transcend/data_sets/OH_squalane_model/geoms/training/*.xyz")[:1000]
filenames.sort()
# energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:100]
energies = np.loadtxt('/Volumes/Transcend/data_sets/vr_ccsd/properties.txt', usecols=[1])[:100]
# energies = np.loadtxt('/Volumes/Transcend/data_sets/OH_squalane_model/geoms/training/properties_kjmol.txt', usecols=[1])[:1000]

estimator.generate_compounds(filenames)
estimator.set_properties(energies)

start_time = time.time()
estimator.generate_descriptor()
end_time = time.time()

print("Generating the descriptor took %s." % (end_time-start_time) )

idx = np.arange(0,len(filenames))

idx_train, idx_test = modsel.train_test_split(idx, test_size=0.3)

# train_scores = []

# for i in range(3):
#
# #     an_estimator = clone(estimator)
# #
# #     an_estimator.fit(idx_train)
# #
# #     score_train = an_estimator.score(idx_train)
# #
# #     train_scores.append(score_train)
#
estimator.fit(idx_train)
y_pred = estimator.predict(idx_train)
estimator.correlation_plot(np.squeeze(y_pred), energies[idx_train])
#
#
# # estimator.plot_cost()
#
# # y_pred = estimator.predict(idx_test)
#
# # estimator.correlation_plot(np.squeeze(y_pred), energies[idx_test])
#
#
# print(train_scores)
