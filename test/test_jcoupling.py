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
import glob

import qml
from qml.ml.representations import generate_jcoupling, generate_jcoupling_symmetric
import os

def calc_angle(a,b,c):

    v1 = a - b
    v2 = c - b

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_angle = np.dot(v1,v2)

    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle

#def calc_cosdihedral(p0,p1,p2,p3):
def calc_cosdihedral(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p1

    b1 = b1/np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    # cos(arctan2(y,x)) == x / sqrt(x**2 + y**2)
    cos_dihedral = x / np.sqrt(x**2 + y**2)

    return cos_dihedral

def decay(R, Rc):
    return 0.5 * (1 + np.cos(np.pi * R / Rc))

def two_body_basis(R, Rs, Rc, eta):
    return np.exp(-eta*(R-Rs)**2) * decay(R, Rc)

def three_body_basis(R, T, Rs, Ts, eta, zeta):
    return np.exp(-eta * (R - Rs)**2)[None,:] * \
            (2 * ((1 + np.cos(T - Ts)) * 0.5) ** zeta)[:,None]

def two_body_coupling(d, idx):
    rep = []
    # two body term between coupling atoms
    for i, idx0 in enumerate(idx):
        for idx1 in idx[i+1:]:
            rep.append(d[idx0, idx1])
    return rep

def two_body_coupling_symmetric(distances, idx, rbasis2_12, rbasis2_13,
        eta2_12, eta2_13):
    rep = []

    # atom 0 to 1 plus atom 2 to 3
    d = distances[idx[0], idx[1]]
    basis = two_body_basis(d, rbasis2_12, np.inf, eta2_12)
    d = distances[idx[2], idx[3]]
    basis += two_body_basis(d, rbasis2_12, np.inf, eta2_12)
    rep.append(basis)

    # atom 0 to 2 and atom 1 to 3
    d = distances[idx[0], idx[2]]
    basis = two_body_basis(d, rbasis2_13, np.inf, eta2_13)
    d = distances[idx[1], idx[3]]
    basis += two_body_basis(d, rbasis2_13, np.inf, eta2_13)
    rep.append(basis)

    # atom 0 to 3
    d = distances[idx[0], idx[3]]
    rep.append(d)

    # atom 1 to 2
    d = distances[idx[1], idx[2]]
    rep.append(d)

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
            arep[ele_idx] += two_body_basis(d, basis, rcut, eta)
        return arep

    nbasis = len(basis)
    natoms = distances.shape[0]
    rep = []
    # two body term between coupling atoms and non-coupling
    for idx0 in idx:
        rep.append(atom_to_other(idx0))
    return rep

def two_body_other_symmetric(distances, idx, z, ele, basis, eta, rcut):
    def atom_to_other(idx0):
        arep = np.zeros((len(ele), nbasis))
        for idx1 in range(natoms):
            if idx1 in idx:
                continue
            d = distances[idx0, idx1]
            if d > rcut:
                continue
            ele_idx = np.where(z[idx1] == ele)[0][0]
            arep[ele_idx] += two_body_basis(d, basis, rcut, eta)
        return arep

    nbasis = len(basis)
    natoms = z.size
    rep = []

    # two body term between coupling atoms and non-coupling
    # atom 0/3 to environment
    arep = atom_to_other(idx[0]) + atom_to_other(idx[3])
    rep.append(arep)

    # atom 1/2 to environment
    arep = atom_to_other(idx[1]) + atom_to_other(idx[2])
    rep.append(arep)

    return rep

def three_body_coupling_coupling(coordinates, idx):
    rep = []
    # three body term between coupling atoms
    for i, idx0 in enumerate(idx):
        for j, idx1 in enumerate(idx[i+1:]):
            for idx2 in idx[i+j+2:]:
                angle = calc_angle(coordinates[idx1], coordinates[idx0], coordinates[idx2])
                # cos(angle) to match how the dihedral must be, but probably doesn't matter
                rep.append(np.cos(angle))

    return rep

def three_body_coupling_coupling_symmetric(coordinates, distances, idx, rbasis, abasis1, abasis2, eta, zeta1, zeta2):

    rep = []

    # atom 0, 1, 2 plus atom 1, 2, 3
    d = distances[idx[0], idx[1]]
    angle = calc_angle(coordinates[idx[0]], coordinates[idx[1]], coordinates[idx[2]])
    pair_rep = three_body_basis(d, angle, rbasis, abasis1, eta, zeta1)
    d = distances[idx[2], idx[3]]
    angle = calc_angle(coordinates[idx[1]], coordinates[idx[2]], coordinates[idx[3]])
    pair_rep += three_body_basis(d, angle, rbasis, abasis1, eta, zeta1)
    rep.append(pair_rep)

    # atom 0, 1, 3 plus atom 0, 2, 3
    d = distances[idx[0], idx[1]]
    angle = calc_angle(coordinates[idx[0]], coordinates[idx[1]], coordinates[idx[3]])
    pair_rep = three_body_basis(d, angle, rbasis, abasis2, eta, zeta2)
    d = distances[idx[2], idx[3]]
    angle = calc_angle(coordinates[idx[0]], coordinates[idx[2]], coordinates[idx[3]])
    pair_rep += three_body_basis(d, angle, rbasis, abasis2, eta, zeta2)
    rep.append(pair_rep)

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
            # Should be centered around one of the coupling atoms
            angle = calc_angle(coordinates[idx1], coordinates[idx0], coordinates[idx2])
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

def three_body_coupling_other_symmetric(coordinates, distances, idx, z, ele, rbasis, abasis, eta, zeta, rcut):
    def pair_to_other(idx0, idx1):
        pair_rep = np.zeros((len(ele), nrbasis, nabasis))
        for idx2 in range(natoms):
            if idx2 in idx:
                continue
            d = (distances[idx0, idx2] + distances[idx1, idx2]) * 0.5
            if d > rcut:
                continue

            ele_idx = np.where(z[idx2] == ele)[0]
            angle = calc_angle(coordinates[idx1], coordinates[idx0], coordinates[idx2])
            pair_rep[ele_idx] += three_body_basis(d, angle, rbasis, abasis, eta, zeta) * \
                    decay(d, rcut)

        return pair_rep

    nrbasis = len(rbasis)
    nabasis = len(abasis)
    natoms = z.size
    rep = []

    # three body term between two coupling atoms and one non-coupling
    # atom 0,1/2,3 to environment
    pair_rep = pair_to_other(idx[0], idx[1])
    pair_rep += pair_to_other(idx[3], idx[2]) # The order here matters
    rep.append(pair_rep)
    # atom 0,2/1,3
    pair_rep = pair_to_other(idx[0], idx[2])
    pair_rep += pair_to_other(idx[3], idx[1])
    rep.append(pair_rep)
    # atom 0,3
    pair_rep = pair_to_other(idx[0], idx[3])
    rep.append(pair_rep)
    # atom 1,2
    pair_rep = pair_to_other(idx[1], idx[2])
    rep.append(pair_rep)

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
            for idx2 in range(idx1+1, natoms):
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

def three_body_other_other_symmetric(coordinates, distances, idx, z, pairs, rbasis, abasis, eta, zeta, rcut):
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
    # atom 0/3
    three_body_rep = atom_to_other_pair(idx[0])
    three_body_rep += atom_to_other_pair(idx[3])
    rep.append(three_body_rep)
    # atom 0/3
    three_body_rep = atom_to_other_pair(idx[1])
    three_body_rep += atom_to_other_pair(idx[2])
    rep.append(three_body_rep)

    return rep

def four_body(coordinates, idx):
    dihedral = calc_cosdihedral(coordinates[idx])
    return [dihedral]


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

        this_representation.append(
                four_body(coordinates, idx))

        all_representations.append(nd_to_1d(this_representation))

    all_representations = np.asarray(all_representations)

    return all_representations



def nd_to_1d(x):
    """
    Ravels a list of lists or floats to a 1d representation
    """

    rep = []

    for item in x:
        if isinstance(item, float):
            rep.append(item)
        else:
            rep.extend(nd_to_1d(item))

    return np.asarray(rep)

def pycoupling_symmetric(coordinates, coupling_idx, nuclear_charges, elements,
        nbasis, precision, cutoff):

    rbasis2 = np.linspace(0.9, cutoff, nbasis)
    rbasis3 = rbasis2
    abasis = np.linspace(0, np.pi, nbasis)
    n_elements = len(elements)
    natoms = len(coordinates)
    nTs = nbasis
    rcut2 = cutoff
    acut = cutoff
    eta2 = precision**2 * np.log(2) / ((cutoff - 0.9) / (nbasis - 1))**2
    eta3 = eta2
    zeta = np.log(2)/(np.log(2) - np.log(1 + np.cos((np.pi / (nbasis - 1))/precision)))

    rbasis2_12 = np.linspace(0.95, 2.0, nbasis)
    rbasis2_13 = np.linspace(1.2,  3.4, nbasis)
    eta2_12 = precision**2 * np.log(2) / ((2.0 - 0.95) / (nbasis - 1))**2
    eta2_13 = precision**2 * np.log(2) / ((3.4 - 1.2) / (nbasis - 1))**2
    abasis_123 = np.linspace(0.75, np.pi, nbasis)
    abasis_124 = np.linspace(0.65, np.pi, nbasis)
    zeta_123 = np.log(2)/(np.log(2) - np.log(1 + np.cos(((np.pi - 0.75) / (nbasis - 1))/precision)))
    zeta_124 = np.log(2)/(np.log(2) - np.log(1 + np.cos(((np.pi - 0.65) / (nbasis - 1))/precision)))

    distances = np.sqrt(np.sum((coordinates[:,None] - coordinates[None,:])**2, axis = 2))

    pairs = []
    for i, el1 in enumerate(elements):
        for el2 in elements[i:]:
            pairs.append([el1,el2])

    pairs = np.asarray(pairs, dtype = int)

    all_representations = []

    for idx in coupling_idx:
        this_representation = []

        print(nd_to_1d(this_representation).size)
        this_representation.append(
                two_body_coupling_symmetric(
                    distances, idx, rbasis2_12, rbasis2_13, 
                    eta2_12, eta2_13))
        print(nd_to_1d(this_representation).size)

        this_representation.append(
                two_body_other_symmetric(
                    distances, idx, nuclear_charges, elements, rbasis2, eta2, rcut2))

        print(nd_to_1d(this_representation).size)
        this_representation.append(
                three_body_coupling_coupling_symmetric(
                    coordinates, distances, idx, rbasis2_12, abasis_123, abasis_124, eta2_12, zeta_123, zeta_124))

        print(nd_to_1d(this_representation).size)
        this_representation.append(
                three_body_coupling_other_symmetric(
                    coordinates, distances, idx, nuclear_charges, elements,
                    rbasis3, abasis, eta3, zeta, acut))

        print(nd_to_1d(this_representation).size)
        this_representation.append(
                three_body_other_other_symmetric(
                    coordinates, distances, idx, nuclear_charges, pairs,
                    rbasis3, abasis, eta3, zeta, acut))
        print(nd_to_1d(this_representation).size)

        this_representation.append(
                four_body(coordinates, idx))
        print(nd_to_1d(this_representation).size)

        all_representations.append(nd_to_1d(this_representation))

    all_representations = np.asarray(all_representations)

    return all_representations


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

    #files = glob.glob("qm7/*.xyz")

    # Joint asymmetric and symmetric
    rcut2 = 5
    rbasis2 = np.linspace(0, rcut2, 3)
    eta2 = 1.1
    rcut3 = 5
    rbasis3 = np.linspace(0, rcut3, 3)
    abasis = np.linspace(0, np.pi, 3)
    eta3 = 0.9
    zeta = 0.8
    # Only symmetric
    rbasis2_12 = np.linspace(0.5, 2.0, 3)
    rbasis2_13 = np.linspace(1.5, 3.0, 3)
    eta2_12 = 1.2
    eta2_13 = 1.15
    abasis_123 = np.linspace(np.pi/2, np.pi, 3)
    abasis_124 = np.linspace(0, np.pi, 3)
    zeta_3 = 0.85

    path = test_dir = os.path.dirname(os.path.realpath(__file__))

    mols = []
    for xyz_file in files:
        mol = qml.data.Compound(xyz=path + "/" + xyz_file)
        mols.append(mol)

    elements = set()
    for mol in mols:
        elements = elements.union(mol.nuclear_charges)

    elements = sorted(list(elements))


    for i, mol in enumerate(mols):
        #fort_rep = generate_jcoupling(mol.nuclear_charges, mol.coordinates, [[5,0,1,2],[6,2,1,0]],
        #        elements, 3, 3, 3, 1.1, 0.9, 0.8, 5, 5)
        #fort_rep = generate_jcoupling(mol.nuclear_charges, mol.coordinates, [[5,0,1,2],[6,2,1,0]],
        #        elements, 3, 2, 5)
        #py_rep = pycoupling(mol.coordinates, [[5,0,1,2],[6,2,1,0]], mol.nuclear_charges,
        #            elements, rbasis2, eta2, rcut2, rbasis3, abasis, eta3, zeta, rcut3)
        py_rep_sym = pycoupling_symmetric(mol.coordinates, [[5,0,1,2],[6,2,1,0]], mol.nuclear_charges,
                        elements, 3, 2, 5)
        fort_rep_sym = generate_jcoupling_symmetric(mol.nuclear_charges, mol.coordinates, [[5,0,1,2],[6,2,1,0]],
                elements, 3, 2, 5)
        start = 26
        end = 27
        print(py_rep_sym[0,start:end])
        print(fort_rep_sym[0,start:end])
        assert(np.allclose(py_rep_sym[:,:end], fort_rep_sym[:,:end]))

def get_parameters():
    files = glob.glob('qm7/*.dat')
    files.extend(glob.glob('/home/lb17101/dev/ML_for_NMR/xyz/*.dat'))

    all_coordinate_pairs = []

    for dat_file in files:
        mol = qml.data.Compound(xyz=dat_file.replace(".dat", ".xyz"))
        with open(dat_file) as f:
            lines = f.readlines()

            dihedral_pairs = []
            for i, line0 in enumerate(lines):
                if "TORSION ANGLES" in line0:
                    for line1 in lines[i+1:]:
                        tokens = line1.split()
                        if len(tokens) == 0:
                            break
                        a0 = int(tokens[0]) - 1
                        a1 = int(tokens[1]) - 1
                        a2 = int(tokens[2]) - 1
                        a3 = int(tokens[3]) - 1
                        if a0 == a3:
                            continue
                        if mol.nuclear_charges[a0] not in [1,6]:
                            continue
                        if mol.nuclear_charges[a1] not in [1,6,7]:
                            continue
                        if mol.nuclear_charges[a2] not in [1,6,7]:
                            continue
                        if mol.nuclear_charges[a3] not in [1,6]:
                            continue
                        dihedral_pairs.append([a0, a1, a2, a3])
            if len(dihedral_pairs) == 0:
                continue
            dihedral_pairs = np.asarray(dihedral_pairs, dtype = int)

        all_coordinate_pairs.append(mol.coordinates[dihedral_pairs])

    all_coordinate_pairs = np.concatenate(all_coordinate_pairs)
    # distances between atom 0 and 1
    d = np.sqrt(np.sum((all_coordinate_pairs[:,0,:] - all_coordinate_pairs[:,1,:])**2, axis=1))
    print("12", d.min(), d.max())
    # distances between atom 0 and 2
    d = np.sqrt(np.sum((all_coordinate_pairs[:,0,:] - all_coordinate_pairs[:,2,:])**2, axis=1))
    print("13", d.min(), d.max())
    # distances between atom 0 and 3
    d = np.sqrt(np.sum((all_coordinate_pairs[:,0,:] - all_coordinate_pairs[:,3,:])**2, axis=1))
    print("14", d.min(), d.max())
    # distances between atom 1 and 2
    d = np.sqrt(np.sum((all_coordinate_pairs[:,1,:] - all_coordinate_pairs[:,2,:])**2, axis=1))
    print("23", d.min(), d.max())
    # distances between atom 1 and 3
    d = np.sqrt(np.sum((all_coordinate_pairs[:,1,:] - all_coordinate_pairs[:,3,:])**2, axis=1))
    print("24", d.min(), d.max())
    # distances between atom 2 and 3
    d = np.sqrt(np.sum((all_coordinate_pairs[:,2,:] - all_coordinate_pairs[:,3,:])**2, axis=1))
    print("34", d.min(), d.max())
    # angle between atom 0 1 2
    angles = []
    for pair in all_coordinate_pairs:
        a = calc_angle(pair[0,:], pair[1,:], pair[2,:])
        angles.append(a)
    angles = np.asarray(angles)
    print("123", angles.min(), angles.max())
    # angle between atom 0 1 3
    angles = []
    for pair in all_coordinate_pairs:
        a = calc_angle(pair[0,:], pair[1,:], pair[3,:])
        angles.append(a)
    angles = np.asarray(angles)
    print("124", angles.min(), angles.max())
    # angle between atom 0 2 3
    angles = []
    for pair in all_coordinate_pairs:
        a = calc_angle(pair[0,:], pair[2,:], pair[3,:])
        angles.append(a)
    angles = np.asarray(angles)
    print("134", angles.min(), angles.max())
    # angle between atom 1 2 3
    angles = []
    for pair in all_coordinate_pairs:
        a = calc_angle(pair[1,:], pair[2,:], pair[3,:])
        angles.append(a)
    angles = np.asarray(angles)
    print("234", angles.min(), angles.max())
    quit()


if __name__ == "__main__":
    test_jcoupling()
    #get_parameters()




