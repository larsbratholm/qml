import pickle
import glob
import numpy as np
from aglaia.wrappers import OAMNN, OMNN
import tensorflow as tf

estimator = OAMNN(batch_size = 30, representation = "slatm", scoring_function='mae', iterations=3000, hl1=30, hl2=20, slatm_dgrid1 = 0.06)
filenames = glob.glob("/Volumes/Transcend/repositories/Aglaia/data/qm7/*.xyz")[:100]
filenames.sort()
energies = np.loadtxt('/Volumes/Transcend/repositories/Aglaia/data/qm7/hof_qm7.txt', usecols=[1])[:100]
estimator.generate_compounds(filenames)
estimator.set_properties(energies)
estimator.generate_descriptor()

pickle.dump(estimator, open('model.pickle', 'wb'))
with open('idx.csv', 'w') as f:
    for i in range(energies.size):
        f.write('%s\n' % i)

    

