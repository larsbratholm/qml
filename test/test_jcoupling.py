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

def decay(R, Rc):
    return 0.5 * (1 + np.cos(np.pi * R / Rc))

def two_body_basis(R, Rs, Rc, eta):
    return np.exp(-eta*(R-Rs)**2) * decay(R, Rc)

def three_body_basis(R, T, Rs, Ts, eta, zeta):
    return np.exp(-eta * (R - Rs)**2)[:,None] * \
            (2 * ((1 + np.cos(T - Ts)) * 0.5) ** zeta)[None,:]


def two_body_coupling(d, idx):
    rep = []
    # two body term between coupling atoms
    for i, idx0 in enumerate(idx):
        for idx1 in idx[i+1:]:
            rep.append(d[idx0, idx1])
    return rep

def two_body_other(distances, idx, z, ele, basis, eta, rcut):
    def atom_to_other(idx0):
        arep = np.zeros((len(ele), nbasis))
        for idx1 in range(natoms):
            if idx1 in idx:
                continue
            d = distances[idx0, idx1]
            if d > rcut:
                continue
            ele_idx = np.where(z[idx1] == ele)[0][0]
            arep[ele_idx] += two_body_basis(d, basis, eta, rcut)
        return arep

    nbasis = len(basis)
    natoms = distances.shape[0]
    rep = []
    # two body term between coupling atoms and non-coupling
    for idx0 in idx:
        rep.append(atom_to_other(idx0))
    return rep

def three_body_coupling_coupling(coordinates, idx):
    rep = []
    # three body term between coupling atoms
    for i, idx0 in enumerate(idx):
        for j, idx1 in enumerate(idx[i+1:]):
            for idx2 in idx[i+j+2:]:
                angle = calc_angle(coordinates[idx1], coordinates[idx0], coordinates[idx2])
                rep.append(angle)

    return rep

def three_body_coupling_other(coordinates, distances, idx, z, ele, rbasis, abasis, eta, zeta, rcut):
    def pair_to_other(idx0, idx1):
        pair_rep = np.zeros((len(ele), nrbasis, nabasis))
        for idx2 in range(natoms):
            if idx2 in idx:
                continue
            d = (distances[idx0, idx2] + distances[idx1, idx2]) * 0.5
            if d > rcut:
                continue

            ele_idx = np.where(z[idx2] == ele)[0]
            angle = calc_angle(coordinates[idx0], coordinates[idx2], coordinates[idx1])
            pair_rep[ele_idx] += three_body_basis(d, angle, rbasis, abasis, eta, zeta) * \
                    decay(d, rcut)

        return pair_rep

    nrbasis = len(rbasis)
    nabasis = len(abasis)
    natoms = z.size

    rep = []
    # three body term between two coupling atoms and one non-coupling
    for i, idx0 in enumerate(idx):
        for idx1 in idx[i+1:]:
            rep.append(pair_to_other(idx0, idx1))

    return rep


def three_body_other_other(coordinates, distances, idx, z, pairs, rbasis, abasis, eta, zeta, rcut):
    def atom_to_other_pair(idx0):
        pair_rep = np.zeros((len(pairs), nrbasis, nabasis))
        for idx1 in range(natoms):
            if idx1 in idx:
                continue
            d1 = distances[idx0, idx1]
            if d1 > rcut:
                continue
            for idx2 in range(natoms):
                if idx2 in idx:
                    continue
                d2 = distances[idx0, idx2]
                if d2 > rcut:
                    continue

                sorted_z = np.sort([z[idx1], z[idx2]])

                pair_idx = np.where((pairs[:,0] == sorted_z[0]) & 
                        (pairs[:,1] == sorted_z[1]))[0][0]

                angle = calc_angle(coordinates[idx1], coordinates[idx0], coordinates[idx2])

                pair_rep[pair_idx] += three_body_basis((d1+d2)/2, angle, rbasis, abasis, eta, zeta) \
                        * decay(d1, rcut) * decay(d2, rcut)

        return pair_rep

    nrbasis = len(rbasis)
    nabasis = len(abasis)
    natoms = z.size

    rep = []
    # three body term between two coupling atoms and one non-coupling
    for i, idx0 in enumerate(idx):
        rep.append(atom_to_other_pair(idx0))

    return rep

def pycoupling(coordinates, coupling_idx, nuclear_charges, elements,
        rbasis2, eta2, rcut2, rbasis3, abasis, eta3, zeta, rcut3):

    distances = np.sqrt(np.sum((coordinates[:,None] - coordinates[None,:])**2, axis = 2))

    pairs = []
    for i, el1 in enumerate(elements):
        for el2 in elements[i:]:
            pairs.append([el1,el2])

    pairs = np.asarray(pairs, dtype = int)

    all_representations = []

    for idx in coupling_idx:
        this_representation = []

        this_representation.append(
                two_body_coupling(
                    distances, idx))

        this_representation.append(
                two_body_other(
                    distances, idx, nuclear_charges, elements, rbasis2, eta2, rcut2))

        this_representation.append(
                three_body_coupling_coupling(
                    coordinates, idx))

        this_representation.append(
                three_body_coupling_other(
                    coordinates, distances, idx, nuclear_charges, elements,
                    rbasis3, abasis, eta3, zeta, rcut3))

        this_representation.append(
                three_body_other_other(
                    coordinates, distances, idx, nuclear_charges, pairs,
                    rbasis3, abasis, eta3, zeta, rcut3))


        for i in this_representation:
            if isinstance(i[0], float):
                print(i)
            else:
                for j in i:
                    print(j)

        quit()





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

    rcut2 = 5
    rbasis2 = np.arange(0, rcut2, 3)
    eta2 = 0.9
    rcut3 = 5
    rbasis3 = np.arange(0.5, rcut3, 3)
    abasis = np.arange(0, np.pi, 3)
    eta3 = 1.1
    zeta = 0.8

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
        pycoupling(mol.coordinates, [[5,0,1,2],[6,2,1,0]], mol.nuclear_charges,
                elements, rbasis2, eta2, rcut2, rbasis3, abasis, eta3, zeta, rcut3)


if __name__ == "__main__":
    test_jcoupling()

