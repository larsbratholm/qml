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

import itertools
import contextlib
from collections import defaultdict
import time
import numpy as np
import os
import qml
from qml.representations import *
from qml.data import NUCLEAR_CHARGE


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
    
    sorted_cm = np.zeros(cm.size)

    for i in range(size):
        si = sortargs[i]
        for j in range(i+1):
            sj = sortargs[j]
            if j < i:
                idx = (i*(i+1))//2 + j
            else:
                idx = (j*(j+1))//2 + i
            if sj < si:
                idy = (si*(si+1))//2 + sj
            else:
                idy = (sj*(sj+1))//2 + si

            sorted_cm[idx] = cm[idy]

    return sorted_cm

def unsorted_coulomb_matrix(nuclear_charges, coordinates, size):

    natoms = nuclear_charges.size
    cm = np.zeros((size, size))
    feature_vector = np.zeros((size*(size+1))//2)
    for i in range(natoms):
        cm[i,i] = 0.5 * nuclear_charges[i] ** 2.4
        idx = (i*(i+1))//2
        feature_vector[idx+i] = cm[i,i]
        for j in range(i):
            d = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
            cm[i,j] = nuclear_charges[i] * nuclear_charges[j] / d
            cm[j,i] = nuclear_charges[i] * nuclear_charges[j] / d
            feature_vector[idx+j] = cm[i,j]

    return cm, feature_vector

def test_atomic_coulomb_matrix():

    size = mol.nuclear_charges.size + 1 # +1 to check that dummy atoms are handled correctly

    # Generate atomic coulomb matrix representation, sorted by distance, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance")

    # Generate atomic coulomb matrix representation, sorted by distance, using the python interface
    acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "distance")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

    # Generate atomic coulomb matrix representation, sorted by distance, with reference implementation
    acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, "distance")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"
    
    # Generate atomic coulomb matrix representation, sorted by row-norm, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm")

    acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "row-norm")

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

    # Generate atomic coulomb matrix representation, sorted by distance,
    # with cutoffs, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0)

    acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "distance",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0)

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

    # Generate atomic coulomb matrix representation, sorted by row-norm,
    # with cutoffs, using the Compound class
    mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0)

    acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "row-norm",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0)

    assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"


    # Generate the sncf1 variant of the atomic coulomb matrix representation, sorted by distance,
    # with cutoffs using the python interface
    acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "distance",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf1", localization = 2.0)

    acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "distance",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf1", localization = 2.0)

    assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    # Generate the sncf1 variant of the atomic coulomb matrix representation, sorted by row-norm,
    # with cutoffs using the python interface
    acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "row-norm",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf1", localization = 2.0)

    acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "row-norm",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf1", localization = 2.0)

    assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    # Generate the sncf2 variant of the atomic coulomb matrix representation, sorted by distance,
    # with cutoffs using the python interface
    acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "distance",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf2", localization = 2.0)

    acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "distance",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf2", localization = 2.0)

    assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    # Generate the sncf2 variant of the atomic coulomb matrix representation, sorted by distance,
    # with cutoffs using the python interface
    acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "row-norm",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf2", localization = 2.0)

    acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
            mol.coordinates, size = size, sorting = "row-norm",
            central_cutoff = 4.0, central_decay = 0.5,
            interaction_cutoff = 5.0, interaction_decay = 1.0,
            variant = "sncf2", localization = 2.0)

    assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

def atomic_coulomb_matrix(nuclear_charges, coordinates, size, sorting = "distance",
        central_cutoff = 1e6, central_decay = -1, interaction_cutoff = 1e6, interaction_decay = -1,
        variant = "classic", localization = 1.0):

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
    sorted_cm = np.zeros((natoms, (size * (size + 1)) // 2))
    cm_mat = np.zeros((natoms, size, size))

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

    if sorting == "row-norm":
        sorting = np.zeros((natoms, natoms))
        for k in range(natoms):
            norms = np.zeros(natoms)
            norms[k] = np.inf
            for i in range(natoms):
                if i == k:
                    continue
                for j in range(natoms):
                    norms[i] += cm_mat[k,i,j]**2

            sorting[k] = np.argsort(norms)[::-1]

    else: # sort by distances
        sorting = np.zeros((natoms, natoms))
        for k in range(natoms):
            distances = np.sum((coordinates - coordinates[k].T)**2, axis = 1)
            sorting[k] = np.argsort(distances)



    for k in range(natoms):
        sortargs = sorting[k]
        idx = 0
        for i in range(natoms):
            si = sortargs[i]
            for j in range(i+1):
                sj = sortargs[j]
                sorted_cm[k, idx] = cm_mat[k, si, sj]
                idx += 1

    return sorted_cm

def test_eigenvalue_coulomb_matrix():

    size = mol.nuclear_charges.size + 1 # +1 to check that dummy atoms are handled correctly

    # Generate coulomb matrix eigenvalue representation using the Compound class
    mol.generate_eigenvalue_coulomb_matrix(size = size)

    # Generate coulomb matrix eigenvalue representation using the python interface
    ecm = generate_eigenvalue_coulomb_matrix(mol.nuclear_charges,
                mol.coordinates, size = size)


    assert np.allclose(mol.representation, ecm), "Error in coulomb matrix eigenvalue representation"

    # Compare with python implementation
    ecm2 = eigenvalue_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size = size)

    assert np.allclose(ecm2, ecm), "Error in coulomb matrix eigenvalue representation"

def test_bob():

    asize = dict([(key, value+1) for key,value in mol.natypes.items()])

    # Generate bag of bonds representation using the Compound class
    mol.generate_bob(asize)

    # Generate bag of bonds representation using the python interface
    bob = generate_bob(mol.nuclear_charges,
            mol.coordinates, mol.atomtypes, asize = asize)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

    # Compare with python implementation
    bob = bob_reference(mol.nuclear_charges, mol.coordinates, mol.atomtypes, size = (mol.natoms), asize = asize)

    assert np.allclose(mol.representation, bob), "Error in bag of bonds representation"

def bob_reference(nuclear_charges, coordinates, atomtypes, size = 23, asize = {"O":3, "C":7, "N":3, "H":16, "S":1}):

    coulomb_matrix = generate_coulomb_matrix(nuclear_charges,
            coordinates, size = size, sorting = "unsorted")

    coulomb_matrix = vector_to_matrix(coulomb_matrix)

    atoms = sorted(asize, key=asize.get)
    nmax = [asize[key] for key in atoms]

    descriptor = []
    positions = dict([(element, np.where(atomtypes == element)[0]) for element in atoms])
    for i, (element1, size1) in enumerate(zip(atoms,nmax)):
        pos1 = positions[element1]
        feature_vector = np.zeros(size1)
        feature_vector[:pos1.size] = np.diag(coulomb_matrix)[pos1]
        descriptor.append(feature_vector)
        for j, (element2, size2) in enumerate(zip(atoms,nmax)):
            if i > j:
                continue
            if i == j:
                size = (size1*(size1-1))//2
                feature_vector = np.zeros(size)
                sub_matrix = coulomb_matrix[np.ix_(pos1,pos1)]
                feature_vector[:(pos1.size*(pos1.size-1))//2] = sub_matrix[np.triu_indices(pos1.size, 1)]
                feature_vector[::-1].sort()
                descriptor.append(feature_vector)
            else:
                pos2 = positions[element2]
                feature_vector = np.zeros(size1*size2)
                feature_vector[:pos1.size*pos2.size] = coulomb_matrix[np.ix_(pos1,pos2)].ravel()
                feature_vector[::-1].sort()
                descriptor.append(feature_vector)

    return np.concatenate(descriptor)

def test_local_bob():

    t = time.time()
    path = test_dir = os.path.dirname(os.path.realpath(__file__))
    mols = []
    for i in range(1,50):
        n = str(i)
        n = "0"*(4-len(n)) + n
        xyz_file = "qm7/%s.xyz" % n
        mol = qml.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)
    print (time.time() - t)

    t = time.time()
    asize = {}
    for mol in mols:
        for key, value in mol.natypes.items():
            if key not in asize.keys():
                asize[key] = value
                continue
            asize[key] = max(asize[key], value)
    print (time.time() - t)
    print (asize)


    t = time.time()
    for mol in mols:
       # print (time.time() - t)
        # Generate atomic coulomb matrix representation, sorted by row-norm, using the Compound class
        if (len(mol.atomtypes) != 8):
            continue
        bob = generate_local_bob(mol.nuclear_charges,
                mol.coordinates, mol.atomtypes, asize = asize)
    print (time.time() - t)

    ## Generate atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs, using the Compound class
    #mol.generate_atomic_coulomb_matrix(size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"

    ## Generate atomic coulomb matrix representation, sorted by row-norm,
    ## with cutoffs, using the Compound class
    #mol.generate_atomic_coulomb_matrix(size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #acm = atomic_coulomb_matrix(mol.nuclear_charges, mol.coordinates, size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0)

    #assert np.allclose(mol.representation, acm), "Error in atomic coulomb matrix representation"


    ## Generate the sncf1 variant of the atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    ## Generate the sncf1 variant of the atomic coulomb matrix representation, sorted by row-norm,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf1", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    ## Generate the sncf2 variant of the atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "distance",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"

    ## Generate the sncf2 variant of the atomic coulomb matrix representation, sorted by distance,
    ## with cutoffs using the python interface
    #acm = generate_atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #acm2 = atomic_coulomb_matrix(mol.nuclear_charges,
    #        mol.coordinates, size = size, sorting = "row-norm",
    #        central_cutoff = 4.0, central_decay = 0.5,
    #        interaction_cutoff = 5.0, interaction_decay = 1.0,
    #        variant = "sncf2", localization = 2.0)

    #assert np.allclose(acm2, acm), "Error in atomic coulomb matrix representation"
    pass

def local_bob_reference(nuclear_charges, coordinates, atomtypes, size = 23, asize = {"O":3, "C":7, "N":3, "H":16, "S":1},
        central_cutoff = 1e6, central_decay = -1, interaction_cutoff = 1e6, interaction_decay = -1):

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
    cm_mat = np.zeros((natoms, size, size))

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
    for i, (element1, size1) in enumerate(zip(atoms,nmax)):
        pos1 = positions[element1]
        # A-bag
        for k in range(natoms):
            feature_vector = np.diag(cm_mat[k])[pos1]
            descriptor[k].append(feature_vector)
        for j, (element2, size2) in enumerate(zip(atoms,nmax)):
            if i > j:
                continue
            if i == j:
                for k in range(natoms):
                    # Ak,Ai-bag
                    feature_vector = np.zeros(size1-1)
                    if atomtypes[k] == element1:
                        pos = pos1[pos1 != k]
                        feature_vector = cm_mat[k,k,pos]

                    descriptor[k].append(feature_vector)

                    # Ai,Aj bag
                    if atomtypes[k] == element1:
                        pos = pos1[pos1 != k]
                        idx1, idx2 = np.triu_indices(size1-1,1)
                        feature_vector = np.zeros((size1*(size1-1))//2)
                        feature_vector[:idx1.size] = cm_mat(k,idx1,idx2).ravel()
                    else:
                        idx1, idx2 = np.triu_indices(size1,1)
                        feature_vector = cm_mat(k,idx1,idx2).ravel()

                    descriptor[k].append(feature_vector)

            else:
                pos2 = positions[element2]

                for k in range(natoms):
                    # Ak,Bi-bag
                    feature_vector = np.zeros(size2)
                    if atomtypes[k] == element1:
                        feature_vector = cm_mat[k,k,pos2]

                    descriptor[k].append(feature_vector)

                    # Ai,Bk-bag
                    feature_vector = np.zeros(size1)
                    if atomtypes[k] == element2:
                        feature_vector = cm_mat[k,k,pos1]

                    descriptor[k].append(feature_vector)

                    # Ai,Bj bag
                    if atomtypes[k] in [element1,element2]:
                        if atomtypes[k] == element1:
                            bagsize = (size1-1)*size2
                        else:
                            bagsize = (size2-1)*size1

                        feature_vector = np.zeros(size1*size2)
                        pos = np.ix_([k], pos1[pos1 != k], pos2[pos2 != k])
                        feature_vector[:bagsize] = cm_mat(pos).ravel()
                    else:
                        idx1, idx2 = np.triu_indices(size1,1)
                        feature_vector = cm_mat(k,idx1,idx2)

                    descriptor[k].append(feature_vector)


    representation = np.empty(natoms,np.concatenate(descriptor[0]).size)

    for i in range(natoms):
        for j in range(len(descriptor(i))):
            descriptor[i][j][::-1].sort()
        representation[i] = np.concatenate(descriptor[i])

    return representation

def vector_to_matrix(vec):
    size = (-1 + int(np.sqrt(1 + 8 * vec.size))) // 2
    mat = np.zeros((size,size))

    count = 0
    for i in range(size):
        for j in range(i+1):
            mat[i,j] = vec[count]
            mat[j,i] = mat[i,j]
            count += 1

    return mat

if __name__ == "__main__":
    #test_coulomb_matrix()
    #test_atomic_coulomb_matrix()
    #test_eigenvalue_coulomb_matrix()
    #test_bob()
    test_local_bob()

