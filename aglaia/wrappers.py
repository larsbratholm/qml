"""
Helper classes for hyper parameter optimization with osprey
"""

#TODO relative imports
from aglaia import _NN, MRMP


class _OSPNN(_NN):
    """
    Adds additional variables and functionality to the _NN class that makes interfacing with
    Osprey for hyperparameter search easier
    """

    def __init__(self, _hl1 = 0, _hl2 = 0, _hl3 = 0, **args):
        """
        :param _hl1: Number of neurons in the first hidden layer. If different from zero, ``hidden_layer_sizes`` is
                    overwritten.
        :type _hl1: integer
        :param _hl2: Number of neurons in the second hidden layer. Ignored if ``_hl1`` is different from zero.
        :type _hl2: integer
        :param _hl3: Number of neurons in the third hidden layer. Ignored if ``_hl1`` or ``_hl2`` is different from zero.
        :type _hl3: integer
        """

        super(_OSPNN, self).__init__(**args)

        self._hl1 = _hl1
        self._hl2 = _hl2
        self._hl3 = _hl3
        self._process_hidden_layers()

        # Placeholder variables
        self.compounds = np.empty(0, dtype=object)
        self.properties = np.empty(0, dtype=float)

    def _process_hidden_layers(self)
        if self._hl1 != 0:
            if self._hl2 in [0, None]:
                size = 1
            elif self._hl3 in [0, None]:
                size = 2
            else:
                size = 3

            # checks on self._hl1
            if not is_positive_integer(self._hl1):
                raise ValueError("Hidden layer size must be a positive integer. Got %s" % str(self._hl1))

            # checks on self._hl2
            if size >= 2:
                if not is_positive_integer(self._hl2):
                    raise ValueError("Hidden layer size must be a positive integer. Got %s" % str(self._hl2))

            # checks on self._hl2
            if size == 3:
                if not is_positive_integer(self._hl3):
                    raise ValueError("Hidden layer size must be a positive integer. Got %s" % str(self._hl3))

            if size == 1:
                self.hidden_layer_sizes = [int(self._hl1)]
            elif size == 2:
                self.hidden_layer_sizes = [int(self._hl1), int(self._hl2)]
            elif size == 3:
                self.hidden_layer_sizes = [int(self._hl1), int(self._hl2), int(self._hl3)]

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
                raise ValueError("Number of properties (%d) does not match number of compounds (%d)" 
                        % (self.properties.size, len(filenames)))


        self.compounds = np.empty(len(filenames), dtype=object)
        for i, filename in enumerate(filenames):
            self.compounds[i] = Compound(filename)

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
        if not is_positive_integer(pad):
            raise ValueError("Expected variable 'pad' to be a positive integer. Got %s" % str(pad))

        asize = {}

        for mol in mols:
            for key, value in mol.natypes.items():
                if key not in asize:
                    asize[key] = value + pad
                    continue
                asize[key] = max(asize[key], value + pad)

        return asize

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
        if not is_positive_integer(pad):
            raise ValueError("Expected variable 'pad' to be a positive integer. Got %s" % str(pad))

        nmax = max(mol.natoms for mol in self.compounds)

        return nmax + pad


# Molecular Representation Single Property
class OSPMRMP(_NN, _OSPNN):
    """
    Adds additional variables and functionality to the MRMP class that makes interfacing with
    Osprey for hyperparameter search easier.
    """

    def __init__(self, representation = 'unsorted_coulomb_matrix', **args):
        """
        A molecule's cartesian coordinates and chemical composition is transformed into a descriptor for the molecule,
        which is then used as input to a single or multi layered feedforward neural network with a single output.
        This class inherits from the _NN and _OSPNN class and all inputs not unique to the MRMP class is passed to the
        parents.

        Available representations at the moment are ['unsorted_coulomb_matrix', 'sorted_coulomb_matrix',
        bag_of_bonds', 'slatm'].

        :param representation: Name of molecular representation.
        :type representation: string

        """

        super(OSPMRMP,self).__init__(**args)

        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise ValueError("Unknown representation %s" % representation)
        self.representation = representation.lower()

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
                raise ValueError("Number of properties (%d) does not match number of compounds (%d)" 
                        % (len(y), self.compounds.size))

        self.properties = np.asarray(y, dtype = float)

    def fit(self, indices):
        """
        Fit the neural network to a set of molecular descriptors and targets. It is assumed that QML compounds and
        properties have been set in advance and which indices to use is given.

        :param indices: Which indices of the pregenerated QML compounds and properties to use.
        :type indices: integer array

        """

        if self.properties.size == 0:
            raise ValueError("Properties needs to be set in advance")
        if self.compounds.size == 0:
            raise ValueError("QML compounds needs to be created in advance")

        if not is_positive_integer_or_zero(indices[0]):
            raise ValueError("Expected input to be indices")

        try:
            idx = np.asarray(indices, dtype=int)
            if not np.array_equal(idx, indices):
                raise ValueError
        except ValueError:
            raise ValueError("Expected input to be indices")

        if self.representation == 'unsorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))/2
            x = np.empty(idx.size, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_coulomb_matrix(size = nmax, sorting = "unsorted")
                x[i] = mol.representation

        if self.representation == 'sorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))/2
            x = np.empty((idx.size, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_coulomb_matrix(size = nmax, sorting = "sorted")
                x[i] = mol.representation

        elif self.representation == "bag_of_bonds":
            asize = self._get_asize()
            representation_size = (nmax*(nmax+1))/2
            X = np.empty((len(x), representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[x]):
                mol.generate_bob(asize = asize)
                X[i] = mol.representation
        elif self.representation == "slatm":
            mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in self.compounds]))
            representation_size = (nmax*(nmax+1))/2
            X = np.empty((len(x), representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[x]):
                mol.generate_bob(asize = asize)
                X[i] = mol.representation
        
        y = self.properties[idx]

        return self._fit(x, y)

