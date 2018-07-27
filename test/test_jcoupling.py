# MIT License
#
# Copyright (c) 2018 Lars Andersen Bratholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import numpy as np

import qml
#from qml.ml.representations import generate_jcoupling
import os

def calc_angle(a,b,c):

    v1 = a - b
    v2 = c - b

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_angle = np.dot(v1,v2)

    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle

def two_body_coupling_coupling(d, idx):
    rep = []
    # two body term between coupling atoms
    for i, idx0 in enumerate(idx):
        for idx1 in idx[i+1:]:
            rep.append(d[idx0, idx1])
    return rep

def two_body_coupling_other(distances, idx, z):
    rep = []
    # two body term between coupling atoms and non-coupling
    for i, idx0 in enumerate(idx):
        for idx1 in idx[i+1:]:
            rep.append(distances[idx0, idx1])
    return rep

def pycoupling(coordinates, coupling_idx, nuclear_charges, elements):

    distances = np.sqrt(np.sum((coordinates[:,None] - coordinates[None,:])**2, axis = 2))

    all_representations = []

    for idx in coupling_idx:
        this_representation = []

        this_representation.append(two_body_coupling_coupling(distances, idx))

        this_representation.append(two_body_coupling_other(distances, idx))





def test_jcoupling():
    files = ["qm7/0101.xyz",
             "qm7/0102.xyz",
             "qm7/0103.xyz",
             "qm7/0104.xyz",
             "qm7/0105.xyz",
             "qm7/0106.xyz",
             "qm7/0107.xyz",
             "qm7/0108.xyz",
             "qm7/0109.xyz",
             "qm7/0110.xyz"]


    path = test_dir = os.path.dirname(os.path.realpath(__file__))

    mols = []
    for xyz_file in files:
        mol = qml.data.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    elements = set()
    for mol in mols:
        elements = elements.union(mol.nuclear_charges)

    elements = list(elements)

    for mol in mols:
        pycoupling(mol.coordinates, [[0,1,2,3],[1,2,4,5]], mol.nuclear_charges,
                elements)


if __name__ == "__main__":
    test_jcoupling()

