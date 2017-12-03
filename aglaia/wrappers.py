"""
Helper classes for hyper parameter optimization with osprey
"""

import glob
import itertools
import numpy as np

#TODO relative imports
from aglaia import _NN, MRMP
from utils import InputError, is_positive_integer, is_string, is_positive_integer_or_zero


class _OSPNN(_NN):
    """
    Adds additional variables and functionality to the _NN class that makes interfacing with
    Osprey for hyperparameter search easier
    """

    def __init__(self, hl1 = 5, hl2 = 0, hl3 = 0, **args):
        """
        :param hl1: Number of neurons in the first hidden layer. If different from zero, ``hidden_layer_sizes`` is
                    overwritten.
        :type hl1: integer
        :param hl2: Number of neurons in the second hidden layer. Ignored if ``hl1`` is different from zero.
        :type hl2: integer
        :param hl3: Number of neurons in the third hidden layer. Ignored if ``hl1`` or ``hl2`` is different from zero.
        :type hl3: integer
        """

        super(_OSPNN, self).__init__(**args)

        self.hl1 = hl1
        self.hl2 = hl2
        self.hl3 = hl3
        self._process_hidden_layers()

        # Placeholder variables
        self.compounds = np.empty(0, dtype=object)
        self.properties = np.empty(0, dtype=float)

    # TODO test
    def _process_hidden_layers(self):
        if self.hl1 == 0:
            raise InputError("hl1 must be larger than zero. Got %s" % str(self.hl1))

        if self.hl2 == 0:
            size = 1
        elif self.hl3 == 0:
            size = 2
        else:
            size = 3

        # checks on self.hl1
        if not is_positive_integer(self.hl1):
            raise InputError("Hidden layer size must be a positive integer. Got %s" % str(self.hl1))

        # checks on self.hl2
        if size >= 2:
            if not is_positive_integer(self.hl2):
                raise InputError("Hidden layer size must be a positive integer. Got %s" % str(self.hl2))

        # checks on self.hl2
        if size == 3:
            if not is_positive_integer(self.hl3):
                raise InputError("Hidden layer size must be a positive integer. Got %s" % str(self.hl3))

        if size == 1:
            self.hidden_layer_sizes = [int(self.hl1)]
        elif size == 2:
            self.hidden_layer_sizes = [int(self.hl1), int(self.hl2)]
        elif size == 3:
            self.hidden_layer_sizes = [int(self.hl1), int(self.hl2), int(self.hl3)]

    #TODO test
    def generate_compounds(self, filenames):
        """
        Creates QML compounds. Needs to be called before fitting.

        :param filenames: path of xyz-files
        :type filenames: list
        """

        try:
            from qml import Compound
        except ModuleNotFoundError:
            raise ModuleNotFoundError("The module qml is required")

        # Check that the number of properties match the number of compounds
        if self.properties.size == 0:
            pass
        else:
            if self.properties.size == len(filenames):
                pass
            else:
                raise InputError("Number of properties (%d) does not match number of compounds (%d)" 
                        % (self.properties.size, len(filenames)))


        self.compounds = np.empty(len(filenames), dtype=object)
        for i, filename in enumerate(filenames):
            self.compounds[i] = Compound(filename)

    # TODO test
    def _get_asize(self, pad = 0):
        """
        Gets the maximum occurrences of each element in a single molecule. To support larger molecules
        an optional padding can be added by the ``pad`` variable.

        :param pad: Add an integer padding to the returned dictionary
        :type pad: integer

        :return: dictionary of the maximum number of occurences of each element in a single molecule.
        :rtype: dictionary

        """

        if self.compounds.size == 0:
            raise RuntimeError("QML compounds have not been generated")
        if not is_positive_integer_or_zero(pad):
            raise InputError("Expected variable 'pad' to be a positive integer or zero. Got %s" % str(pad))

        asize = {}

        for mol in self.compounds:
            for key, value in mol.natypes.items():
                if key not in asize:
                    asize[key] = value + pad
                    continue
                asize[key] = max(asize[key], value + pad)

        return asize

    # TODO test
    def _get_msize(self, pad = 0):
        """
        Gets the maximum number of atoms in a single molecule. To support larger molecules
        an optional padding can be added by the ``pad`` variable.

        :param pad: Add an integer padding to the returned dictionary
        :type pad: integer

        :return: largest molecule with respect to number of atoms.
        :rtype: integer

        """

        if self.compounds.size == 0:
            raise RuntimeError("QML compounds have not been generated")
        if not is_positive_integer_or_zero(pad):
            raise InputError("Expected variable 'pad' to be a positive integer or zero. Got %s" % str(pad))

        nmax = max(mol.natoms for mol in self.compounds)

        return nmax + pad

# Molecular Representation Single Property
class OSPMRMP(MRMP, _OSPNN):
    """
    Adds additional variables and functionality to the MRMP class that makes interfacing with
    Osprey for hyperparameter search easier.
    """

    def __init__(self, representation = 'unsorted_coulomb_matrix', 
            slatm_sigma1 = 0.05, slatm_sigma2 = 0.05, slatm_dgrids1 = 0.03, slatm_dgrids2 = 0.03, rpower = 6,
            **args):
        """
        A molecule's cartesian coordinates and chemical composition is transformed into a descriptor for the molecule,
        which is then used as input to a single or multi layered feedforward neural network with a single output.
        This class inherits from the _NN and _OSPNN class and all inputs not unique to the MRMP class is passed to the
        parents.

        Available representations at the moment are ['unsorted_coulomb_matrix', 'sorted_coulomb_matrix',
        bag_of_bonds', 'slatm'].

        :param representation: Name of molecular representation.
        :type representation: string
        :param slatm_sigma1: Scale of the gaussian bins for the two-body term
        :type slatm_sigma1: float
        :param slatm_sigma2: Scale of the gaussian bins for the three-body term
        :type slatm_sigma2: float
        :param slatm_dgrids1: Spacing between the gaussian bins for the two-body term
        :type slatm_dgrids1: float
        :param slatm_dgrids2: Spacing between the gaussian bins for the three-body term
        :type slatm_dgrids2: float

        """

        super(OSPMRMP,self).__init__(**args)

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation = representation.lower()

    # TODO test
    def set_properties(self, y):
        """
        Set properties. Needed to be called before fitting.

        :param y: array of properties of size (nsamples,)
        :type y: array
        """

        if self.compounds.size == 0:
            pass
        else:
            if self.compounds.size == len(y):
                pass
            else:
                raise InputError("Number of properties (%d) does not match number of compounds (%d)" 
                        % (len(y), self.compounds.size))

        self.properties = np.asarray(y, dtype = float)

    # TODO test
    def fit(self, indices):
        """
        Fit the neural network to a set of molecular descriptors and targets. It is assumed that QML compounds and
        properties have been set in advance and which indices to use is given.

        :param indices: Which indices of the pregenerated QML compounds and properties to use.
        :type indices: integer array

        """

        if self.properties.size == 0:
            raise InputError("Properties needs to be set in advance")
        if self.compounds.size == 0:
            raise InputError("QML compounds needs to be created in advance")

        if not is_positive_integer_or_zero(indices[0]):
            raise InputError("Expected input to be indices")

        try:
            idx = np.asarray(indices, dtype=int)
            if not np.array_equal(idx, indices):
                raise InputError
        except InputError:
            raise InputError("Expected input to be indices")

        if self.representation == 'unsorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((idx.size, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_coulomb_matrix(size = nmax, sorting = "unsorted")
                x[i] = mol.representation

        if self.representation == 'sorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((idx.size, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_coulomb_matrix(size = nmax, sorting = "row-norm")
                x[i] = mol.representation

        elif self.representation == "bag_of_bonds":
            asize = self._get_asize()
            x = np.empty(idx.size, dtype=object)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_bob(asize = asize)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        elif self.representation == "slatm":
            from qml.representations import get_slatm_mbtypes
            mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
            x = np.empty(idx.size, dtype=object)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_slatm(mbtypes)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)
#     unit_cell=None, local=False, sigmas=[0.05,0.05], dgrids=[0.03,0.03],
#     rcut=4.8, alchemy=False, pbc='000', rpower=6):

        
        y = self.properties[idx]

        return
        return self._fit(x, y)

if __name__ == "__main__":
    import time
    for rep in ["unsorted_coulomb_matrix", "sorted_coulomb_matrix", "bag_of_bonds", "slatm"]:
        x = OSPMRMP(representation=rep)
        filenames = glob.glob("/home/lab/dev/qml/tests/qm7/*.xyz")[:100]
        y = np.array(range(len(filenames)), dtype=int)
        x.generate_compounds(filenames)
        x.set_properties(y)
        t = time.time()
        x.fit(y)
        print(rep, t - time.time())

