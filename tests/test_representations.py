# MIT License
#
# Copyright (c) 2017 Anders Steen Christensen, Lars Andersen Bratholm
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

import time
import numpy as np
import os
import qml
from qml.representations import *
from qml.data import NUCLEAR_CHARGE

def get_asize(mols, pad):

    asize = {}

    for mol in mols:
        for key, value in mol.natypes.items():
            if key not in asize:
                asize[key] = value + pad
                continue
            asize[key] = max(asize[key], value + pad)
    return asize

def test_representations():
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
        mol = qml.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    size = max(mol.nuclear_charges.size for mol in mols) + 1

    asize = get_asize(mols,1)

    coulomb_matrix(mols, size, path)
    atomic_coulomb_matrix(mols, size, path)
    eigenvalue_coulomb_matrix(mols, size, path)
    bob(mols, asize, path)

def coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_coulomb_matrix(size = size, sorting = "row-norm")

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"

    # Generate coulomb matrix representation, unsorted, using the Compound class
    for i, mol in enumerate(mols): 
        mol.generate_coulomb_matrix(size = size, sorting = "unsorted")

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/coulomb_matrix_representation_unsorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"

def atomic_coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by distance
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance")

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_distance_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"
    # Compare to old implementation (before 'indices' keyword)
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_distance_sorted_no_indices.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"


    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm")

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by distance, with soft cutoffs
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance",
                central_cutoff = 4.0, central_decay = 0.5,
                interaction_cutoff = 5.0, interaction_decay = 1.0)

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_distance_sorted_with_cutoff.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by row-norm, with soft cutoffs
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm",
                central_cutoff = 4.0, central_decay = 0.5,
                interaction_cutoff = 5.0, interaction_decay = 1.0)

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/atomic_coulomb_matrix_representation_row-norm_sorted_with_cutoff.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate only two atoms in the coulomb matrix representation, sorted by distance
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance")
        representation_subset = mol.representation[1:3]
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance", indices = [1,2])
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i,j] - mol.representation[i,j]
                if abs(diff) > 1e-9:
                    print (i,j,diff, representation_subset[i,j],mol.representation[i,j])
        assert np.allclose(representation_subset, mol.representation), \
                "Error in atomic coulomb matrix representation"

    # Generate only two atoms in the coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm")
        representation_subset = mol.representation[1:3]
        mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm", indices = [1,2])
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i,j] - mol.representation[i,j]
                if abs(diff) > 1e-9:
                    print (i,j,diff, representation_subset[i,j],mol.representation[i,j])
        assert np.allclose(representation_subset, mol.representation), \
                "Error in atomic coulomb matrix representation"

def eigenvalue_coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_eigenvalue_coulomb_matrix(size = size)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/eigenvalue_coulomb_matrix_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in eigenvalue coulomb matrix representation"

def bob(mols, asize, path):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols): 
        mol.generate_bob(asize)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(path + "/data/bob_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in bag of bonds representation"

def print_mol(mol):
    n = len(mol.representation.shape)
    if n == 1:
        for item in mol.representation:
            print("{:.9e}".format(item), end='  ')
        print()
    elif n == 2:
        for atom in mol.representation:
            for item in atom:
                print("{:.9e}".format(item), end='  ')
            print()

def test_local_bob():

    path = test_dir = os.path.dirname(os.path.realpath(__file__))
    base = "qm9/dsgdb9nsd_"
    #base = "qm7/"
    nlen = 6
    nmol = 50
    mols = []
    for i in range(1,nmol):
        n = str(i)
        n = "0"*(nlen-len(n)) + n
        xyz_file = base + "%s.xyz" % n
        try:
            mol = qml.Compound(xyz=path + "/" + xyz_file)
            mols.append(mol)
        except IOError:
            pass

    asize = {}
    for mol in mols:
        for key, value in mol.natypes.items():
            if key not in asize.keys():
                asize[key] = value
                continue
            asize[key] = max(asize[key], value)

    t = time.time()
    for mol in mols:
       # print (time.time() - t)
        # Generate atomic coulomb matrix representation, sorted by row-norm, using the Compound class
        bob = generate_local_bob(mol.nuclear_charges,
                mol.coordinates, mol.atomtypes, asize = asize, 
                central_cutoff = 2.0, central_decay = 1.0, interaction_cutoff = 3.0, interaction_decay = 0.5)
        bob2 = local_bob_reference(mol.nuclear_charges, mol.coordinates, mol.atomtypes, asize, 
                central_cutoff = 2.0, central_decay = 1.0, interaction_cutoff = 3.0, interaction_decay = 0.5)

        assert(np.allclose(bob,bob2))

        bob = generate_local_bob(mol.nuclear_charges,
                mol.coordinates, mol.atomtypes, asize = asize, variant = "sncf1",
                central_cutoff = 2.0, central_decay = 1.0, interaction_cutoff = 3.0, interaction_decay = 0.5)
        bob2 = local_bob_reference(mol.nuclear_charges, mol.coordinates, mol.atomtypes, asize, variant = "sncf1",
                central_cutoff = 2.0, central_decay = 1.0, interaction_cutoff = 3.0, interaction_decay = 0.5)

        bob = generate_local_bob(mol.nuclear_charges,
                mol.coordinates, mol.atomtypes, asize = asize, variant = "sncf2",
                central_cutoff = 2.0, central_decay = 1.0, interaction_cutoff = 3.0, interaction_decay = 0.5)
        bob2 = local_bob_reference(mol.nuclear_charges, mol.coordinates, mol.atomtypes, asize, variant = "sncf2",
                central_cutoff = 2.0, central_decay = 1.0, interaction_cutoff = 3.0, interaction_decay = 0.5)


        for i in range(bob.shape[0]):
            for j in range(bob.shape[1]):
                #if (i == 0 and j in [3,4,5]):
                #    print (bob[i,j], bob[i,j])
                diff = bob[i,j] - bob2[i,j]
                if abs(diff) > 1e-9:
                    print(i+1,j+1,diff,bob[i,j],bob2[i,j])

        assert(np.allclose(bob,bob2))


    print (time.time() - t)

def local_bob_reference(nuclear_charges, coordinates, atomtypes, asize = {"O":3, "C":7, "N":3, "H":16, "S":1},
        central_cutoff = 1e6, central_decay = -1, interaction_cutoff = 1e6, interaction_decay = -1, variant = "classic",
        localization = 1):

    if central_cutoff < 0:
        central_cutoff = 1e6

    if interaction_cutoff < 0 or interaction_cutoff > 2 * central_cutoff:
        interaction_cutoff = 2 * central_cutoff

    if central_decay < 0:
        central_decay = 0
    elif central_decay > central_cutoff:
        central_decay = central_cutoff

    if interaction_decay < 0:
        interaction_decay = 0
    elif interaction_decay > interaction_cutoff:
        interaction_decay = interaction_cutoff

    natoms = nuclear_charges.size
    cm_mat = np.zeros((natoms, natoms, natoms))

    for k in range(natoms):
        for i in range(natoms):
            for j in range(i, natoms):
                if variant == "classic" and i == j:
                    dik = np.sqrt(np.sum((coordinates[i] - coordinates[k])**2))
                    if dik < central_cutoff:
                        cm_mat[k,i,i] = 0.5 * nuclear_charges[i]**2.4
                        if dik > central_cutoff - central_decay:
                            cm_mat[k,i,i] *= (0.5 * (1 + np.cos(np.pi * (dik - central_cutoff + central_decay) / central_decay)))**2
                elif i == j and j == k:
                    dik = np.sqrt(np.sum((coordinates[i] - coordinates[k])**2))
                    if dik < central_cutoff:
                        cm_mat[k,i,i] = 0.5 * nuclear_charges[i]**2.4
                        if dik > central_cutoff - central_decay:
                            cm_mat[k,i,i] *= (0.5 * (1 + np.cos(np.pi * (dik - central_cutoff + central_decay) / central_decay)))**2
                else:
                    dij = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                    if dij < interaction_cutoff:
                        dik = np.sqrt(np.sum((coordinates[i] - coordinates[k])**2))
                        djk = np.sqrt(np.sum((coordinates[j] - coordinates[k])**2))
                        if dik < central_cutoff and djk < central_cutoff:
                            cm_mat[k,i,j] = nuclear_charges[i] * nuclear_charges[j] / \
                            (dij*(variant in ["classic","sncf2"]) + (dik+djk) * (variant in ["sncf1","sncf2"]))**localization

                            if dij > interaction_cutoff - interaction_decay:
                                cm_mat[k,i,j] *= 0.5 * (1 + np.cos(np.pi * (dij - interaction_cutoff + interaction_decay) / interaction_decay))
                            if dik > central_cutoff - central_decay:
                                cm_mat[k,i,j] *= 0.5 * (1 + np.cos(np.pi * (dik - central_cutoff + central_decay) / central_decay))
                            if djk > central_cutoff - central_decay:
                                cm_mat[k,i,j] *= 0.5 * (1 + np.cos(np.pi * (djk - central_cutoff + central_decay) / central_decay))

                    cm_mat[k,j,i] = cm_mat[k,i,j]

    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]
    descriptor = [[] for _ in atomtypes]
    positions = dict([(element, np.where(atomtypes == element)[0]) for element in atoms])

    # X-bag
    for k in range(natoms):
        descriptor[k].append(np.asarray([cm_mat[k,k,k]]))

    # A-bag
    if variant == "classic":
        for i, (element1, size1) in enumerate(zip(atoms,nmax)):
            pos1 = positions[element1]
            for k in range(natoms):
                pos = pos1[pos1 != k]
                feature_vector = np.diag(cm_mat[k])[pos]
                feature_vector = np.pad(feature_vector, (size1-feature_vector.size,0), "constant")
                descriptor[k].append(feature_vector)

    # XA bags
    for i, (element1, size1) in enumerate(zip(atoms,nmax)):
        pos1 = positions[element1]

        for k in range(natoms):
            pos = pos1[pos1 != k]
            feature_vector = cm_mat[k,k,pos]
            feature_vector = np.pad(feature_vector, (size1-feature_vector.size,0), "constant")
            descriptor[k].append(feature_vector)

    # AB bags
    for i, (element1, size1) in enumerate(zip(atoms,nmax)):
        pos1 = positions[element1]
        for j, (element2, size2) in enumerate(zip(atoms,nmax)):
            if i > j:
                continue
            if i == j:
                for k in range(natoms):
                    pos = pos1[pos1 != k]

                    offset = 0
                    if variant == "classic":
                        offset = 1
                    idx1, idx2 = np.triu_indices(pos.size,offset)
                    feature_vector = np.zeros((size1*(size1+1-2*offset))//2)
                    feature_vector[:idx1.size] = cm_mat[k,pos[idx1],pos[idx2]].ravel()

                    descriptor[k].append(feature_vector)

            else:
                pos2 = positions[element2]

                for k in range(natoms):
                    pos1_no_k = pos1[pos1 != k]
                    pos2_no_k = pos2[pos2 != k]
                    bagsize = pos1_no_k.size * pos2_no_k.size
                    feature_vector = np.zeros(size1 * size2)
                    pos = np.ix_([k], pos1_no_k, pos2_no_k)
                    feature_vector[:bagsize] = cm_mat[pos].ravel()
                    descriptor[k].append(feature_vector)

    representation = np.empty((natoms,np.concatenate(descriptor[0]).size))

    for i in range(natoms):
        for j in range(len(descriptor[i])):
            descriptor[i][j][::-1].sort()
        representation[i] = np.concatenate(descriptor[i])

    return representation


if __name__ == "__main__":
    test_local_bob()

