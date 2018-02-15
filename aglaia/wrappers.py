"""
Helper classes for hyper parameter optimization with osprey
"""

import glob
import itertools
from inspect import signature
import numpy as np
from sklearn.base import BaseEstimator

#TODO relative imports
from .aglaia import _NN, MRMP
from .utils import InputError, is_positive_integer, is_string, is_positive_integer_or_zero, \
        is_non_zero_integer, is_bool, is_positive


class _OSPNN(BaseEstimator, _NN):
    """
    Adds additional variables and functionality to the _NN class that makes interfacing with
    Osprey for hyperparameter search easier
    """

    def __init__(self, hl1 = 5, hl2 = 0, hl3 = 0,
            compounds = None, properties = None, nuclear_charges = None, coordinates = None, **kwargs):
        """
        :param hl1: Number of neurons in the first hidden layer. If different from zero, ``hidden_layer_sizes`` is
                    overwritten.
        :type hl1: integer
        :param hl2: Number of neurons in the second hidden layer. Ignored if ``hl1`` is different from zero.
        :type hl2: integer
        :param hl3: Number of neurons in the third hidden layer. Ignored if ``hl1`` or ``hl2`` is different from zero.
        :type hl3: integer
        """

        super(_OSPNN, self).__init__(**kwargs)
        self._set_hl(hl1, hl2, hl3)

        # Placeholder variables
        self.compounds = np.empty(0, dtype=object)
        self.properties = np.empty(0, dtype=float)

        if type(compounds) != type(None):
            self.compounds = compounds
        if type(properties) != type(None):
            self.properties = properties

    def get_params(self, deep = True):
        """
        Hack that overrides the get_params routine of BaseEstimator.
        self.get_params() returns the input parameters of __init__. However it doesn't
        handle inheritance well, as we would like to include the input parameters to
        __init__ of all the parents as well.

        :returns: params - dictionary of parameters and their user-set value

        """
        # This gets the params of OSPMRMP and puts them in a dictionary 'params'
        params = BaseEstimator.get_params(self)

        # This gets the parameters from _NN
        grandparent_init = super(_OSPNN, self).__init__
        grandparent_init_signature = signature(grandparent_init)

        parameters_nn = (p for p in grandparent_init_signature.parameters.values()
                         if p.name != 'self' and p.kind != p.VAR_KEYWORD)

        for p in parameters_nn:
            if p.name in params:
                return InputError('This should never happen')

            if hasattr(self, p.name):
                params[p.name] = getattr(self, p.name)
            else:
                params[p.name] = p.default

        # Adding the parameters from _OSPNN, but leaving kwargs out
        parent_init = _OSPNN.__init__
        parent_init_signature = signature(parent_init)

        parameters_ospnn = []
        for p in parent_init_signature.parameters.values():
            if p.name != 'self' and p.kind != p.VAR_KEYWORD:
                if p.name not in params:
                    parameters_ospnn.append(p)

        for p in parameters_ospnn:
            if p.name in params:
                return InputError('This should never happen')

            if p.name == 'kwargs':
                continue

            if hasattr(self, p.name):
                params[p.name] = getattr(self, p.name)
            else:
                params[p.name] = p.default

        return params

    def set_params(self, **params):
        """
        Hack that overrides the set_params routine of BaseEstimator.

        """

        for key, value in params.items():
            key, delim, sub_key = key.partition('__')

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        # recreate hidden_layers_sizes
        self._set_hl(self.hl1, self.hl2, self.hl3)
        return self


    # TODO test
    def _set_hl(self, hl1, hl2, hl3):
        if hl1 == 0:
            raise InputError("hl1 must be larger than zero. Got %s" % str(hl1))

        if hl2 == 0:
            size = 1
        elif hl3 == 0:
            size = 2
        else:
            size = 3

        # checks on hl1
        if not is_positive_integer(hl1):
            raise InputError("Hidden layer size must be a positive integer. Got %s" % str(hl1))

        # checks on hl2
        if size >= 2:
            if not is_positive_integer(hl2):
                raise InputError("Hidden layer size must be a positive integer. Got %s" % str(hl2))

        # checks on hl2
        if size == 3:
            if not is_positive_integer(hl3):
                raise InputError("Hidden layer size must be a positive integer. Got %s" % str(hl3))

        if size == 1:
            self.hidden_layer_sizes = np.asarray([hl1], dtype = int)
        elif size == 2:
            self.hidden_layer_sizes = np.asarray([hl1, hl2], dtype = int)
        elif size == 3:
            self.hidden_layer_sizes = np.asarray([hl1, hl2, hl3], dtype = int)

        self.hl1 = hl1
        self.hl2 = hl2
        self.hl3 = hl3

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

    def score(self, indices):
        idx = np.asarray(indices, dtype=int)
        y = self.properties[idx]
        return self._score(idx, y)


#TODO slatm exception tests
# Molecular Representation Single Property
class OSPMRMP(MRMP, _OSPNN):
    """
    Adds additional variables and functionality to the MRMP class that makes interfacing with
    Osprey for hyperparameter search easier.
    """

    def __init__(self, representation = 'unsorted_coulomb_matrix', 
            slatm_sigma1 = 0.05, slatm_sigma2 = 0.05, slatm_dgrid1 = 0.03, slatm_dgrid2 = 0.03, slatm_rcut = 4.8, slatm_rpower = 6,
            slatm_alchemy = False, compounds = None, properties = None, **kwargs):
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
        :param slatm_dgrid1: Spacing between the gaussian bins for the two-body term
        :type slatm_dgrid1: float
        :param slatm_dgrid2: Spacing between the gaussian bins for the three-body term
        :type slatm_dgrid2: float
        :param slatm_rcut: Cutoff radius
        :type slatm_rcut: float
        :param slatm_rpower: exponent of the binning
        :type slatm_rpower: integer
        :param slatm_alchemy: Whether to use the alchemy version of slatm or not.
        :type slatm_alchemy: bool

        """

        super(OSPMRMP,self).__init__(compounds = compounds, properties = properties, **kwargs)

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation = representation.lower()

        if not is_positive(slatm_sigma1):
            raise InputError("Expected positive float for variable 'slatm_sigma1'. Got %s." % str(slatm_sigma1))
        self.slatm_sigma1 = float(slatm_sigma1)

        if not is_positive(slatm_sigma2):
            raise InputError("Expected positive float for variable 'slatm_sigma2'. Got %s." % str(slatm_sigma2))
        self.slatm_sigma2 = float(slatm_sigma2)

        if not is_positive(slatm_dgrid1):
            raise InputError("Expected positive float for variable 'slatm_dgrid1'. Got %s." % str(slatm_dgrid1))
        self.slatm_dgrid1 = float(slatm_dgrid1)

        if not is_positive(slatm_dgrid2):
            raise InputError("Expected positive float for variable 'slatm_dgrid2'. Got %s." % str(slatm_dgrid2))
        self.slatm_dgrid2 = float(slatm_dgrid2)

        if not is_positive(slatm_rcut):
            raise InputError("Expected positive float for variable 'slatm_rcut'. Got %s." % str(slatm_rcut))
        self.slatm_rcut = float(slatm_rcut)

        if not is_non_zero_integer(slatm_rpower):
            raise InputError("Expected non-zero integer for variable 'slatm_rpower'. Got %s." % str(slatm_rpower))
        self.slatm_rpower = int(slatm_rpower)

        if not is_bool(slatm_alchemy):
            raise InputError("Expected boolean value for variable 'slatm_alchemy'. Got %s." % str(slatm_alchemy))
        self.slatm_alchemy = bool(slatm_alchemy)

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

    def get_representation(self, indices):

        if self.properties.size == 0:
            raise InputError("Properties needs to be set in advance")
        if len(self.compounds) == 0:
            raise InputError("QML compounds needs to be created in advance")

        if not is_positive_integer_or_zero(indices[0]):
            raise InputError("Expected input to be indices")

        try:
            idx = np.asarray(indices, dtype=int)
            if not np.array_equal(idx, indices):
                raise InputError
            # convert to 1d
            idx = idx.ravel()
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
                mol.generate_slatm(mbtypes, local = False, sigmas = [self.slatm_sigma1, self.slatm_sigma2],
                        dgrids = [self.slatm_dgrid1, self.slatm_dgrid2], rcut = self.slatm_rcut, alchemy = self.slatm_alchemy,
                        rpower = self.slatm_rpower)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        return x

    # TODO test
    def fit(self, indices, y = None):
        """
        Fit the neural network to a set of molecular descriptors and targets. It is assumed that QML compounds and
        properties have been set in advance and which indices to use is given.

        :param y: Dummy for osprey
        :type y: None
        :param indices: Which indices of the pregenerated QML compounds and properties to use.
        :type indices: integer array

        """
        x = self.get_representation(indices)

        idx = np.asarray(indices, dtype = int).ravel()
        y = self.properties[idx]

        return self._fit(x, y)

    def predict(self, indices):
        x = self.get_representation(indices)
        return self._predict(x)

if __name__ == "__main__":
    import time
    for rep in ["unsorted_coulomb_matrix", "sorted_coulomb_matrix", "bag_of_bonds", "slatm"]:
        x = OSPMRMP(representation=rep)
        filenames = glob.glob("/home/lb17101/dev/qml/tests/qm7/*.xyz")[:100]
        y = np.array(range(len(filenames)), dtype=int)
        x.generate_compounds(filenames)
        x.set_properties(y)
        t = time.time()
        x.fit(y)
        print(rep, time.time() - t)

