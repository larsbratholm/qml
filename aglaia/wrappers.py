"""
Helper classes for hyper parameter optimization with osprey
"""

import glob
import itertools
from inspect import signature
import numpy as np
from sklearn.base import BaseEstimator
import aglaia.symm_funct as sf
import tensorflow as tf
import aglaia.tests.tensormol_symm_funct as tm_sf

try:
    from qml import Compound, representations
except ModuleNotFoundError:
    raise ModuleNotFoundError("The module qml is required")

from .aglaia import _NN, NN, ARMP
from .utils import InputError, is_array_like, is_numeric_array, is_positive_integer, is_positive_integer_or_zero, \
        is_non_zero_integer, is_positive_integer_or_zero_array, is_dict, is_none, is_string, is_positive, is_bool

class _ONN(BaseEstimator, _NN):
    """
    Adds additional variables and functionality to the _NN class that makes interfacing with
    Osprey for hyperparameter search easier
    """

    def __init__(self, hl1 = 5, hl2 = 0, hl3 = 0,
            compounds = None, properties = None, descriptor=None, zs=None, nuclear_charges = None, coordinates = None, **kwargs):
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
        self.set_descriptor(descriptor)
        self.set_zs(zs)

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

    def set_descriptor(self, descriptor):
        if type(descriptor) != type(None):
            self.descriptor = descriptor
        else:
            self.descriptor = None

    def set_zs(self, zs):
        if type(zs) != type(None):
            self.zs = zs
        else:
            self.zs = None

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

    def _set_acsf(self, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta):
        """
        This function sets the parameters for the acsf as described in the Tensormol paper.

        :param radial_cutoff: float
        :param angular_cutoff: float
        :param radial_rs: np array of floats of shape (n_rad_rs,)
        :param angular_rs:  np array of floats of shape (n_ang_rs,)
        :param theta_s: np array of floats of shape (n_theta_s,)
        :param zeta: float
        :param eta: float
        :return: None
        """

        if not is_positive(radial_cutoff):
            raise InputError("Expected positive float for variable 'radial_cutoff'. Got %s." % str(radial_cutoff))
        self.radial_cutoff = float(radial_cutoff)

        if not is_positive(angular_cutoff):
            raise InputError("Expected positive float for variable 'angular_cutoff'. Got %s." % str(angular_cutoff))
        self.angular_cutoff = float(angular_cutoff)

        if not is_numeric_array(radial_rs):
            raise InputError("Expecting an array like radial_rs. Got %s." % (radial_rs) )
        if not len(radial_rs)>0:
            raise InputError("No radial_rs values were given." )
        self.radial_rs = list(radial_rs)

        if not is_numeric_array(angular_rs):
            raise InputError("Expecting an array like angular_rs. Got %s." % (angular_rs) )
        if not len(angular_rs)>0:
            raise InputError("No angular_rs values were given." )
        self.angular_rs = list(angular_rs)

        if not is_numeric_array(theta_s):
            raise InputError("Expecting an array like theta_s. Got %s." % (theta_s) )
        if not len(theta_s)>0:
            raise InputError("No theta_s values were given. " )
        self.theta_s = list(theta_s)

        if is_numeric_array(eta):
            raise InputError("Expecting a scalar value for eta. Got %s." % (eta))
        self.eta = eta

        if is_numeric_array(zeta):
            raise InputError("Expecting a scalar value for zeta. Got %s." % (zeta))
        self.zeta = zeta

#TODO slatm exception tests
# TODO remove compounds argument and only keep it in _ONN
# Osprey molecular neural network, molecular properties
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

    def get_descriptors_from_indices(self, indices):

        n_samples = indices.shape[0]

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
            x = np.empty((n_samples, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds):
                mol.generate_coulomb_matrix(size = nmax, sorting = "unsorted")
                x[i] = mol.representation

        elif self.representation == 'sorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((n_samples, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds):
                mol.generate_coulomb_matrix(size = nmax, sorting = "row-norm")
                x[i] = mol.representation

        elif self.representation == "bag_of_bonds":
            asize = self._get_asize()
            x = np.empty(n_samples, dtype=object)
            for i, mol in enumerate(self.compounds):
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

        else:

            raise InputError("This should never happen. Unrecognised representation. Got %s." % str(self.representation))

        self.descriptor = x

    def get_representation(self, indices):
        """
        This function takes as input a list of indices and it returns a n_samples by n_features matrix containing all
        the descriptors of the samples that are specified by the indices.
        :param indices: list
        :return: np array of shape (n_samples, n_features)
        """

        if self.properties.size == 0:
            raise InputError("Properties needs to be set in advance")
        if len(self.compounds) == 0:
            raise InputError("QML compounds needs to be created in advance")
        if len(self.descriptor) == 0:
            raise InputError("Descriptors needs to be created in advance")

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

        x_desc = self.descriptor[idx, :]

        return x_desc

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

# Osprey atomic neural network, atomic properties
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

# Osprey atomic neural network, molecular properties
class OAMNN(ARMP, _ONN):
    """
    This class is the wrapper for the ARMP neural network for osprey.
    """
    def __init__(self, representation = 'atomic_coulomb_matrix',
            slatm_sigma1 = 0.05, slatm_sigma2 = 0.05, slatm_dgrid1 = 0.03, slatm_dgrid2 = 0.03, slatm_rcut = 4.8, slatm_rpower = 6,
            slatm_alchemy = False, radial_cutoff=10.0, angular_cutoff=10.0, radial_rs=(0.0, 0.1, 0.2), angular_rs=(0.0, 0.1, 0.2), theta_s=(3.0, 2.0), zeta=3.0,
                 eta=2.0, compounds = None, properties = None, **kwargs):
        """
        A molecule's cartesian coordinates and chemical composition is transformed into a descriptor for the molecule,
        which is then used as input to a single or multi layered feedforward neural network with a single output.
        This class inherits from the ARMP and _ONN class and all inputs not unique to the OAMNN class is passed to the
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
        :param compounds: Contains informations about xyz, zs, ... of all the samples in the data
        :type compounds: list of qml Compounds objects
        :param properties: Molecular properties for all data samples
        :type properties: list
        """

        # TODO try to avoid directly passing compounds and properties. That shouldn't be needed.
        super(OAMNN,self).__init__(compounds = compounds, properties = properties, **kwargs)

        # self._set_representation(representation, slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut,
        #         slatm_rpower, slatm_alchemy)

        self._set_representation(representation)

        if self.representation == 'atomic_coulomb_matrix':
            print("Atomic Coulomb matrix not implemented yet. ")
        if self.representation == 'slatm':
            self._set_slatm(slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut, slatm_rpower, slatm_alchemy)
        if self.representation == 'acsf':
            self._set_acsf(radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)

    def _set_representation(self, representation):
        """
        This function sets the parameter that says which descriptor is going to be used. It also sets the slatm
        parameters (even if the slatm representation isn't going to be used??)

        :param representation: Name of descriptor to use (string)
        :param args: slatm parameters: slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut,
                slatm_rpower, slatm_alchemy.
        :return: None
        """

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['atomic_coulomb_matrix', 'slatm', 'acsf']:
            raise InputError("Unknown representation %s" % representation)
        self.representation = representation.lower()

    def _set_properties(self, properties):
        """
        Sets the properties. Needs to be called before fitting.

        :param y: array of properties of size (n_samples,)
        :type y: array
        """
        if not is_none(properties):
            if is_numeric_array(properties) and np.asarray(properties).ndim == 1:
                self.properties = np.asarray(properties)
            else:
                raise InputError('Variable "properties" expected to be array like of dimension 1. Got %s' % str(properties))
        else:
            self.properties = None

    def generate_descriptor(self):
        """
        This function makes the descriptors for all the compounds.

        :return: None
        """

        if is_none(self.properties):
            raise InputError("Properties needs to be set in advance")
        if is_none(self.compounds):
            raise InputError("QML compounds needs to be created in advance")

        descriptor = None
        zs = None

        if self.representation == 'slatm':

            descriptor, zs = self._generate_slatm()

        elif self.representation == 'atomic_coulomb_matrix':

            print("I still haven't implemented the atomic coulomb matrix. Use slatm or acsf for now.")

        elif self.representation == 'acsf':

            descriptor, zs = self._generate_acsf()

        else:

            raise InputError("This should never happen. Unrecognised representation. Got %s." % str(self.representation))

        self.descriptor = descriptor
        self.zs = zs

    def get_descriptors_from_indices(self, indices):
        """
        This function returns the descriptors that are stored in the compounds for the samples corresponding to the
        indices.

        :param indices: numpy array of shape (n_samples,) of type int
        :return: None
        """

        if is_none(self.properties):
            raise InputError("Properties needs to be set in advance")
        if is_none(self.compounds):
            raise InputError("QML compounds needs to be created in advance")
        if is_none(self.descriptor):
            raise InputError("The descriptors have to be made before being used.")

        if not is_positive_integer_or_zero_array(indices):
            raise InputError("Expected input to be indices")

        # Convert to 1d
        idx = np.asarray(indices, dtype=int).ravel()

        if self.representation == 'slatm':

            return self.descriptor[idx], self.zs[idx]

        elif self.representation == 'atomic_coulomb_matrix':

            print("I still haven't implemented the atomic coulomb matrix. Use slatm for now.")
            return None

        elif self.representation == 'acsf':

            return self.descriptor[idx], self.zs[idx]

        else:

            raise InputError("This should never happen. Unrecognised representation. Got %s." % str(self.representation))

    def _generate_slatm(self):
        """
        This function generates the local slatm descriptor for all the compounds.

        :return: numpy array of shape (n_samples, n_max_atoms, n_features) and (n_samples, n_atoms)
        """

        mbtypes = representations.get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
        list_descriptors = []
        max_n_atoms = 0

        # Generating the descriptor in the shape that ARMP requires it
        for compound in self.compounds:
            compound.generate_slatm(mbtypes, local=True, sigmas=[self.slatm_sigma1, self.slatm_sigma2],
                                    dgrids=[self.slatm_dgrid1, self.slatm_dgrid2], rcut=self.slatm_rcut,
                                    alchemy=self.slatm_alchemy,
                                    rpower=self.slatm_rpower)
            descriptor = compound.representation
            if max_n_atoms < descriptor.shape[0]:
                max_n_atoms = descriptor.shape[0]
            list_descriptors.append(descriptor)

        # Padding the descriptors of the molecules that have fewer atoms
        n_samples = len(list_descriptors)
        n_features = list_descriptors[0].shape[1]
        padded_descriptors = np.zeros((n_samples, max_n_atoms, n_features))
        for i, item in enumerate(list_descriptors):
            padded_descriptors[i, :item.shape[0], :] = item

        # Generating zs in the shape that ARMP requires it
        zs = np.zeros((n_samples, max_n_atoms))
        for i, mol in enumerate(self.compounds):
            zs[i, :mol.nuclear_charges.shape[0]] = mol.nuclear_charges

        return padded_descriptors, zs

    def _generate_acsf(self):
        """
        This function generates the atom centred symmetry functions.
        :return: numpy array of shape (n_samples, n_max_atoms, n_features) and (n_samples, n_atoms)
        """

        # Obtaining the total elements and the element pairs
        mbtypes = representations.get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])

        elements = []
        element_pairs = []

        # Splitting the one and two body interactions in mbtypes
        for item in mbtypes:
            if len(item) == 1:
                elements.append(item[0])
            if len(item) == 2:
                element_pairs.append(list(item))
            if len(item) == 3:
                break

        # Need the element pairs in descending order for TF
        for item in element_pairs:
            item.reverse()

        # Obtaining the xyz and the nuclear charges
        xyzs = []
        zs = []
        max_n_atoms=0

        for compound in self.compounds:
            xyzs.append(compound.coordinates)
            zs.append(compound.nuclear_charges)
            if len(compound.nuclear_charges) > max_n_atoms:
                max_n_atoms = len(compound.nuclear_charges)

        # Padding so that all the samples have the same shape
        n_samples = len(zs)
        for i in range(n_samples):
            current_n_atoms = len(zs[i])
            missing_n_atoms = max_n_atoms - current_n_atoms
            zs_padding = np.zeros(missing_n_atoms)
            zs[i] = np.concatenate((zs[i], zs_padding))
            xyz_padding = np.zeros((missing_n_atoms, 3))
            xyzs[i] = np.concatenate((xyzs[i], xyz_padding))

        zs = np.asarray(zs, dtype=np.int32)
        xyzs = np.asarray(xyzs, dtype=np.float32)

        # # Uncomment to get memory and compute time in tensorboard
        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        # Turning the quantities into tensors
        with tf.name_scope("Inputs"):
            zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
            xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

        # descriptor = sf.generate_parkhill_acsf(xyzs=xyz_tf, Zs=zs_tf, elements=elements, element_pairs=element_pairs,
        #                                        radial_cutoff=self.radial_cutoff, angular_cutoff=self.angular_cutoff,
        #                                        radial_rs=self.radial_rs, angular_rs=self.angular_rs, theta_s=self.theta_s,
        #                                        eta=self.eta, zeta=self.zeta)

        # # Uncomment to use the Parkhill implementation of ACSF
        descriptor = tm_sf.tensormol_acsf(xyz_tf, zs_tf, elements=elements, element_pairs=element_pairs,
                                               radial_cutoff=self.radial_cutoff, angular_cutoff=self.angular_cutoff,
                                               radial_rs=self.radial_rs, angular_rs=self.angular_rs, theta_s=self.theta_s,
                                               eta=self.eta, zeta=self.zeta)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # descriptor_np = sess.run(descriptor, feed_dict={xyz_tf: xyzs, zs_tf: zs})
        # # Uncomment to get memory and compute time in tensorboard
        descriptor_np = sess.run(descriptor, feed_dict={xyz_tf: xyzs, zs_tf: zs}, options=options, run_metadata=run_metadata)
        summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)
        summary_writer.add_run_metadata(run_metadata=run_metadata, tag="Descriptor", global_step=None)
        sess.close()

        return descriptor_np, zs

    def fit(self, indices, zs = None, y = None):
        """
        Fits the neural network to a set of atomic descriptors specified by the indices.
        :param indices: numpy array of indices of shape (n_samples,) of type int
        :param y: Dummy parameter for osprey
        :param zs: Dummy parameter for osprey
        :return: None
        """

        if type(self.descriptor) == type(None):
            self.generate_descriptor()
            print("Generating descriptor...")

        x, zs = self.get_descriptors_from_indices(indices)
        idx = np.asarray(indices, dtype=int).ravel()
        y = self.properties[idx]

        self._fit(x, zs, y)

    def predict(self, indices):
        """
        This function calculates the neural network predictions of the properties for the samples specified by the
        indices.
        :param indices: numpy array of shape (n_samples,) of int
        :return: numpy array of shape (n_samples,) of floats
        """
        x, zs = self.get_descriptors_from_indices(indices)
        y_pred = self._predict([x, zs])
        return y_pred
