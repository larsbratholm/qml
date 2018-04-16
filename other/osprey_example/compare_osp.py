"""
This script overfits 100 data points either from the QM7 data set or from the CH4+CN data sat
"""

from aglaia import aglaia
import time
import numpy as np
from sklearn import model_selection as modsel
import joblib

start_time = time.time()

# data = np.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_qm7.npz")
# descriptors = data["arr_0"]
# zs = data["arr_1"]
# energies = data["arr_2"]

# data = joblib.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_qm7_light.bz")
data = joblib.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_ch4cn_light.bz")
descriptors = data['descriptor']
zs = data["zs"]
energies = data["energies"]

print("Loading the data took", str(time.time() - start_time), " to run")

energies_col = np.reshape(energies, (energies.shape[0], 1))


estimator = aglaia.ARMP(iterations=3000, batch_size=30, scoring_function="mae",
                        hidden_layer_sizes=[30, 20], learning_rate=0.01, tensorboard=True)

xyz_train, xyz_test, zs_train, zs_test, y_train, y_test = modsel.train_test_split(descriptors, zs, energies, test_size=0.2, random_state=42)
# xyz_train, xyz_test, zs_train, zs_test, y_train, y_test = modsel.train_test_split(descriptors, test_zs, energies, test_size=0.2)

start_time_2 = time.time()

# estimator.fit(xyz_train, zs_train, y_train)

print("Training took", str(time.time() - start_time), " to run")

# estimator.save_nn()
estimator.load_nn()

y_pred_test = estimator.predict([xyz_test, zs_test])
y_pred_train = estimator.predict([xyz_train, zs_train])

estimator.correlation_plot(y_pred_train, y_train)

mae = estimator.score([xyz_train, zs_train], y_train)
print("The MAE is " + str(mae))