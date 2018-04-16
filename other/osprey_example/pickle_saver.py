import pickle
import glob
import numpy as np
from aglaia.wrappers import OAMNN

estimator = OAMNN(iterations=400, batch_size=100, scoring_function="mae", tensorboard=False,
                        hidden_layer_sizes=[50, 20], learning_rate=0.005, representation="slatm", slatm_dgrid1 = 0.6)

# filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:100]
filenames = glob.glob("/Volumes/Transcend/data_sets/vr_ccsd/*.xyz")[:100]
filenames.sort()
# energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:100]
energies = np.loadtxt('/Volumes/Transcend/data_sets/vr_ccsd/properties.txt', usecols=[1])[:100]
estimator.generate_compounds(filenames)
estimator.set_properties(energies)
estimator.generate_descriptor()

pickle.dump(estimator, open('model.pickle', 'wb'))
with open('idx.csv', 'w') as f:
    for i in range(energies.size):
        f.write('%s\n' % i)

    

