from aglaia import wrappers
import numpy as np
import glob
from sklearn import model_selection as modsel
from sklearn.base import clone

estimator = wrappers.OAMNN(iterations=400, batch_size=70, scoring_function="mae", tensorboard=True, l2_reg=0.03906198710720926,
                        hidden_layer_sizes=[50, 20], learning_rate=0.005, representation="slatm", slatm_dgrid1 = 0.6)

# filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:100]
filenames = glob.glob("/Volumes/Transcend/data_sets/vr_ccsd/*.xyz")[:100]
filenames.sort()
# energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:100]
energies = np.loadtxt('/Volumes/Transcend/data_sets/vr_ccsd/properties.txt', usecols=[1])[:100]
estimator.generate_compounds(filenames)
estimator.set_properties(energies)
estimator.generate_descriptor()

idx = np.arange(0,len(filenames))

idx_train, idx_test = modsel.train_test_split(idx, test_size=0.3)

train_scores = []

for i in range(3):

    an_estimator = clone(estimator)

    an_estimator.fit(idx_train)

    score_train = an_estimator.score(idx_train)

    train_scores.append(score_train)




# estimator.plot_cost()

# y_pred = estimator.predict(idx_test)

# estimator.correlation_plot(np.squeeze(y_pred), energies[idx_test])


print(train_scores)
