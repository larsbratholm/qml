"""
Helper classes for hyper parameter optimization with osprey
"""

import glob
import itertools
from inspect import signature
import numpy as np
from sklearn.base import BaseEstimator

try:
    from qml import Compound
except ModuleNotFoundError:
    raise ModuleNotFoundError("The module qml is required")

from .aglaia import _NN, NN
from .utils import InputError, is_array_like, is_numeric_array, is_positive_integer, is_positive_integer_or_zero, \
        is_non_zero_integer, is_positive_integer_or_zero_array, is_dict, is_none, is_string, is_positive, is_bool

class _ONN(BaseEstimator, _NN):
    """
    Adds additional variables and functionality to the _NN class that makes interfacing with
    Osprey for hyperparameter search easier
    """

    def __init__(self, hl1 = 5, hl2 = 0, hl3 = 0,
            compounds = None, properties = None, **kwargs):
        """
        :param hl1: Number of neurons in the first hidden layer. If different from zero, ``hidden_layer_sizes`` is
                    overwritten.
        :type hl1: integer
        :param hl2: Number of neurons in the second hidden layer. Ignored if ``hl1`` is different from zero.
        :type hl2: integer
        :param hl3: Number of neurons in the third hidden layer. Ignored if ``hl1`` or ``hl2`` is different from zero.
        :type hl3: integer
        """

        super(_ONN, self).__init__(**kwargs)

        self._set_hl(hl1, hl2, hl3)
        self.set_compounds(compounds)
        self.set_properties(properties)

    def set_compounds(self, compounds):
        self._set_compounds(compounds)

    def _set_compounds(self, compounds):

        if not is_none(compounds):
            if is_array_like(compounds) and isinstance(compounds[0], Compound):
                self.compounds = compounds
            else:
                raise InputError('Variable "compounds" needs to an array of QML compounds. Got %s' % str(compounds))
        else:
            self.compounds = None

    def set_properties(self, properties):
        self._set_properties(properties)

    def _set_properties(self, properties):
        if not is_none(properties):
            self.properties = properties
        else:
            self.properties = None


    # TODO remove class specific things / clean up
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
        grandparent_init = super(_ONN, self).__init__
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

        # Adding the parameters from _ONN, but leaving kwargs out
        parent_init = _ONN.__init__
        parent_init_signature = signature(parent_init)

        parameters_onn = []
        for p in parent_init_signature.parameters.values():
            if p.name != 'self' and p.kind != p.VAR_KEYWORD:
                if p.name not in params:
                    parameters_onn.append(p)

        for p in parameters_onn:
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

        # Check that the number of properties match the number of compounds
        if is_none(self.properties):
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
    
    def _get_slatm_mbtypes(self, arr):
        from qml.representations import get_slatm_mbtypes
        return get_slatm_mbtypes(arr)

    def score(self, indices):
        idx = np.asarray(indices, dtype=int)
        y = self.properties[idx]

        # Osprey maximises a score per default, so return minus mae/rmsd and plus r2
        if self.scoring_function == "r2":
            return self._score(idx, y)
        else:
            return - self._score(idx, y)


#TODO slatm exception tests
# TODO remove compounds argument and only keep it in _ONN
# Osprey molecular neural network
class OMNN(NN, _ONN):
    """
    Adds additional variables and functionality to the NN class that makes interfacing with
    Osprey for hyperparameter search easier.
    Used for generating molecular descriptors to predict global properties, such as energies.
    """

    def __init__(self, representation = 'unsorted_coulomb_matrix', 
            slatm_sigma1 = 0.05, slatm_sigma2 = 0.05, slatm_dgrid1 = 0.03, slatm_dgrid2 = 0.03, slatm_rcut = 4.8, slatm_rpower = 6,
            slatm_alchemy = False, compounds = None, properties = None, **kwargs):
        """
        A molecule's cartesian coordinates and chemical composition is transformed into a descriptor for the molecule,
        which is then used as input to a single or multi layered feedforward neural network with a single output.
        This class inherits from the _NN and _ONN class and all inputs not unique to the OMNN class is passed to the
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

        # TODO try to avoid directly passing compounds and properties. That shouldn't be needed.
        super(OMNN,self).__init__(compounds = compounds, properties = properties, **kwargs)

        self._set_representation(representation, slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut,
                slatm_rpower, slatm_alchemy)

    def _set_properties(self, properties):
        """
        Set properties. Needed to be called before fitting.

        :param y: array of properties of size (nsamples,)
        :type y: array
        """
        if not is_none(properties):
            if is_numeric_array(properties) and np.asarray(properties).ndim == 1:
                self.properties = np.asarray(properties)
            else:
                raise InputError('Variable "properties" expected to be array like of dimension 1. Got %s' % str(properties))
        else:
            self.properties = None

    def _set_representation(self, representation, *args):

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation = representation.lower()

        self._set_slatm(*args)

    def _set_slatm(self, slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut,
            slatm_rpower, slatm_alchemy):

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

    def get_descriptors_from_indices(self, indices):

        if is_none(self.properties):
            raise InputError("Properties needs to be set in advance")
        if is_none(self.compounds):
            raise InputError("QML compounds needs to be created in advance")

        if not is_positive_integer_or_zero_array(indices):
            raise InputError("Expected input to be indices")

        # Convert to 1d
        idx = np.asarray(indices, dtype=int).ravel()

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
            mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
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

        x = self.get_descriptors_from_indices(indices)

        idx = np.asarray(indices, dtype = int).ravel()
        y = self.properties[idx]

        return self._fit(x, y)

    def predict(self, indices):
        x = self.get_descriptors_from_indices(indices)
        return self._predict(x)

# Osprey atomic neural network
class OANN(OMNN):
    """
    Adds additional variables and functionality to the NN class that makes interfacing with
    Osprey for hyperparameter search easier.
    Used for generating atomic descriptors to predict local properties, such as chemical shieldings or j-couplings.
    """

    def __init__(self, representation = 'atomic_coulomb_matrix', 
            slatm_sigma1 = 0.05, slatm_sigma2 = 0.05, slatm_dgrid1 = 0.03, slatm_dgrid2 = 0.03, slatm_rcut = 4.8, slatm_rpower = 6,
            slatm_alchemy = False, compounds = None, properties = None, cm_cutoff = 1e6, **kwargs):
        """
        A descriptor is generated for a single atom or a set of atoms from the carteesian coordinates and chemical
        composition of its environment.
        This is then used as input to a single or multi layered feedforward neural network with a single output.
        This class inherits from the OMNN class and all inputs not unique to the OANN class is passed to the
        parents.

        Available representations at the moment are ['atomic_coulomb_matrix', 'slatm'].

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

        # TODO remove compounds and properties super
        super(OANN,self).__init__(compounds = compounds, properties = properties, representation = representation, **kwargs)

        self._set_cm(cm_cutoff)

    def _set_cm(self, cm_cutoff):
        if is_positive(cm_cutoff):
            self.cm_cutoff = float(cm_cutoff)
        else:
            raise InputError("Expected positive float for variable cm_cutoff. Got %s" % str(cm_cutoff))

    # TODO test
    # TODO check if this actually works with osprey
    # TODO there must be a prettier way of handling data that I'm not seeing.
    #      A single large array would be slower if one needs to look up the compound every time.
    def _set_properties(self, properties):
        """
        Set properties. Needed to be called before fitting.
        `y` needs to be a dictionary with keys corresponding to compound indices.
        Every value is a tuple of two arrays. The first array specifies the properties and the second array
        specifies the indices of the atoms where the property arises from.
        For example the following indicates that for the compound with index 0, the atoms with indices 0 and 1 have
        a property of value 1.2 and 2.3 respectively:

            y[0] = ([0, 1], [1.2, 2.3])

        Multi-index properties is supported. Three-bond coupling constants will directly propagate through four atoms
        and one could have the following, where four atom indices correspond to a single property.

            y[0] = ([[0, 1, 2, 3], [1, 2, 3, 4]], [1.2, 2.3])

        :param y: Dictionary with keys corresponding to compound indices.
        :type y: dictionary
        """
        if not is_none(properties):
            # Check that properties follows the correct format
            if is_dict(properties) and 0 in properties and len(properties[0]) == 2 \
                    and is_array_like(properties[0][0]) and is_array_like(properties[0][1]):
                self.properties = properties
            else:
                raise InputError('Variable "properties" expected to be dictionary. Got %s' % str(properties))
        else:
            self.properties = None

    # This will be run when initialising _OANN
    def _set_representation(self, representation, *args):

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['atomic_coulomb_matrix', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation = representation.lower()

        self._set_slatm(*args)

    def get_descriptors_from_indices(self, indices):
        """
        Constructs the descriptors from the given compounds and indices. Each entry of indices contain the
        index of a compound.
        """

        if is_none(self.properties):
            raise InputError("Properties needs to be set in advance")
        if is_none(self.compounds):
            raise InputError("QML compounds needs to be created in advance")

        if not is_positive_integer_or_zero_array(indices):
            raise InputError("Expected input to be indices")

        # convert to 1d
        idx = np.asarray(indices, dtype=int).ravel()

        if idx.max() >= self.compounds.size:
            raise InputError("Index %d larger than number of given compounds (%d)" % (idx.max(), self.compounds.size))

        x = []

        if self.representation == 'atomic_coulomb_matrix':

            nmax = self._get_msize()
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_atomic_coulomb_matrix(size = nmax, sorting = "distance", central_cutoff = self.cm_cutoff)
                x_i = self._get_descriptors_from_mol(mol, i)
                x.extend(x_i)

        elif self.representation == "slatm":
            mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_slatm(mbtypes, local = True, sigmas = [self.slatm_sigma1, self.slatm_sigma2],
                        dgrids = [self.slatm_dgrid1, self.slatm_dgrid2], rcut = self.slatm_rcut, alchemy = self.slatm_alchemy,
                        rpower = self.slatm_rpower)
                x_i = self._get_descriptors_from_mol(mol, i)
                x.extend(x_i)

        return np.asarray(x)

    def _get_descriptors_from_mol(self, mol, idx):
        """
        Gets the needed atomic descriptors from the molecule. If more that one atom contribute directly to the property
        the atomic representations of them are merged together to a single one.

        """

        indices, _ = self.properties[idx]

        # convert to arrays
        indices_array = np.asarray(indices, dtype = int)

        x = []

        # if the indices array is 1d we just have to match the atomic representation with the respective property
        if indices_array.ndim == 1:
            for i in indices_array:
                x.append(mol.representation[i])

        # if not we have to merge the atomic representations of the atoms in use
        else:
            for i, arr in enumerate(indices_array):
                merged = []
                for j in arr:
                    merged.extend(mol.representation[j])
                x.append(merged[:])

        return x

    def _get_properties_from_indices(self, indices):

        # convert to 1d
        idx = np.asarray(indices, dtype=int).ravel()

        y = []

        for i in indices:
            y_i = list(self.properties[i][1])
            y.extend(y_i)

        return np.asarray(y)


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

        x = self.get_descriptors_from_indices(indices)

        y = self._get_properties_from_indices(indices)

        return self._fit(x, y)

    def predict(self, indices):
        x = self.get_descriptors_from_indices(indices)
        return self._predict(x)
