from aglaia import wrappers
import glob
import numpy as np

m = wrappers.OANN(representation='slatm', iterations=10)
filenames = glob.glob("/home/lb17101/dev/aglaia/examples/qm7_hyperparam_search/qm7/*.xyz")[:100]
m.generate_compounds(filenames)
d = {}
for i, c in enumerate(m.compounds):
    x = []
    y = []
    for j in range(len(c.nuclear_charges)):
        y.append(np.random.random())
        x.append(j)
    d[i] = x[:], y[:]

m.set_properties(d)
x = m.get_descriptors_from_indices([0,1])
y = m._get_properties_from_indices([0,1])
m.fit([0,1])
