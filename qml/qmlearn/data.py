# MIT License
#
# Copyright (c) 2018-2019 Lars Andersen Bratholm
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

import glob
import inspect
import copy

import numpy as np

from ..utils import NUCLEAR_CHARGE, is_string, is_1d_array, is_integer_array, \
        is_numeric_array, is_positive_integer_array

#TODO:
# Check all different things in dicts
# Don't be lazy with docs
# Don't be lazy with checks
# Do tests
# Finish setting properties in xyz reader
# Set natoms in coordinates and nuclear charges, and check that they don't change

class Data(object):
    """
    Scikit-learn compatible Data class. Different file readers / Dataset objects 
    should inherit from this to maintain compatibility.
    """

    def __init__(self):

        self.ncompounds = 0
        self.coordinates = np.empty(0)
        self.nuclear_charges = np.empty(0, dtype=int)
        self.natoms = np.empty(0, dtype=int)
        self.shape = (0,)
        self._indices = np.empty(0, dtype=int)
        self._properties = {'coordinates': 'Atomic',
                           'nuclear_charges': 'Atomic'}

        # Overwritten in various parts of a standard prediction pipeline
        # so don't use these within the class
        self._has_transformed_labels = None
        self._representations = None
        self._kernel = None
        self._representation_type = None
        self._representation_short_name = None
        self._representation_cutoff = None
        self._representation_alchemy = None

    def _set_ncompounds(self, n):
        """
        Updates the number of compounds stored in the class
        """
        if self.ncompounds == 0:
            self.ncompounds = n
            # Hack for sklearn CV
            self.shape = (n,)
            self._indices = np.arange(n)
        else:
            if self.ncompounds != n:
                raise SystemExit("Mismatch in number of compounds stored in the Data object. \
                                 This error suggests that the length of `coordinates`, properties etc. is inconsistent.")

    def take(self, i, axis=None):
        """
        Hack for sklearn CV to make it believe that the
        Data object is a numpy array. Instead of returning
        the intended slice, create a shallow copy of the data
        class and update the _indices and ncompounds attributes.
        We want to have access to the full dataset during the
        pipeline, so we can't take slices of all attributes
        at this point.
        """
        other = copy.copy(self)
        other._indices = i
        other.ncompounds = len(i)
        return other

    def _get_attribute_subset(self, name):
        """
        Returns a subset of an attribute, with the subset
        being specified by self._indices. This should be
        used in the qmlearn pipeline.
        """

        if name in self._properties and hasattr(self, name):
            return getattr(self, name)[self._indices]
        else:
            raise SystemExit("Attribute named %s does not exist." % name)

    def get_attribute(self, name):
        """
        Returns an attribute, given `name`.
        `self.get_attribute(name)` is equivalent to
        `self.name`.

        """

        if name in self._properties and hasattr(self, name):
            return getattr(self, name)
        else:
            raise SystemExit("Attribute named %s does not exist." % name)

    def add_molecular_property(self, data, name):
        """
        Adds a molecular property to the Data class.
        :param data: Array with molecular properties. Can be multi-dimensional.
        :type data: Array
        :param name: Name of the property. The property can be accessed as self.get_attribute(name)
            or self.name
        :type name: Array
        """

        if not is_string(name):
            raise SystemExit("Expected argument `name` to be string. Got %s" % type(name))

        # These should be redundant, but no reason not to check both.
        if name in self._properties or hasattr(self, name):
            raise SystemExit("Property named %s already exists" % name)

        # Check that array is numeric. Also checks if the array is an object array, which
        # only should occur when the number of molecular properties differs between compounds.
        if not is_numeric_array(data):
            raise SystemExit("Expected `data` to be numeric array. Either the content is non-numeric, \
                              or the number of molecular properties differs between compounds.")

        data = np.asarray(data)

        # Note we might want to remove this check in the future,
        # if we ever want to add matrix molecular properties
        if data.ndim > 2:
            raise SystemExit("Expected molecular properties to one dimensional. I.e. arrays, not matrices.")

        # Expand array to explicitly include the dimensionality of
        # the molecular property
        if data.ndim == 1:
            data = data[:,None]

        self._set_ncompounds(data.shape[0])

        setattr(self, name, data)
        self._properties[name] = 'Molecular'

    def add_atomic_property(self, data, name):
        """
        Adds a atomic property to the Data class.
        :param data: Array with atomic properties. Can be multi-dimensional.
        :type data: Array
        :param name: Name of the property. The property can be accessed as self.get_attribute(name)
            or self.name
        :type name: Array
        """

        if not is_string(name):
            raise SystemExit("Expected argument `name` to be string. Got %s" % type(name))

        # These should be redundant, but no reason not to check both.
        if name in self._properties or hasattr(self, name):
            raise SystemExit("Property named %s already exists" % name)

        # Data must be either a numeric array or an object array where
        # each element is an array of size (?,) or (?,n), where n is constant
        # but ? can vary.
        if is_numeric_array(data):
            data = np.asarray(data)
            if data.ndim < 2:
                raise SystemExit("Unexpected shape of argument `data`.")
            # Expand array to explicitly include the dimensionality of
            # the atomic property
            if data.ndim == 2:
                data = data[:,:,None]
            # NOTE: we might want to remove this check in the future,
            # if we ever want to add matrix atomic properties
            elif data.ndim > 3:
                raise SystemExit("Expected each atomic property to be one dimensional. I.e. arrays not matrices.")
        else:
            try:
                # Convert data to object array
                data = np.asarray([np.asarray(mol) for mol in data])

                # If all compounds have the same size, data will no longer
                # be an object array, and we'll pass the array back to the function
                # to be accepted in the former if statement
                if data.dtype != "object":
                    return self.add_atomic_properties(data, name)

                # Check that array is numeric
                if not all(is_numeric_array(mol) for mol in data):
                    raise ValueError

            except ValueError:
                raise SystemExit("Expected `data` to be numeric array.")

            try:
                # Simple way to check that there's the same number of atomic properties
                shapes = np.asarray([mol.shape for mol in data], dtype=int)

                # Check that all properties have the same size
                if (shapes[:,1] != shapes[0,1]).any():
                    raise ValueError

            except ValueError:
                raise SystemExit("Expected atomic properties to have the same shape for all compounds")

            # NOTE: we might want to remove this check in the future,
            # if we ever want to add matrix atomic properties
            if shapes.ndim > 2:
                raise SystemExit("Expected each atomic property to be one dimensional. I.e. arrays not matrices.")

            # Expand array to explicitly include the dimensionality of
            # the atomic property if necessary.
            if shapes[0,1] == 1:
                data = np.asarray([np.asarray(mol)[:,None] for mol in data], dtype=object)

        self._set_ncompounds(data.shape[0])

        setattr(self, name, data)
        self._properties[name] = 'Atomic'

    def set_coordinates(self, coordinates):
        """
        Sets the coordinates attribute with the given `coordinates`.
        :param coordinates: Array with coordinates.
        :type coordinates: Array
        """

        # coordinates must be a numeric array or an object array where
        # each element is an array of size (?,3), where ? can vary.
        if is_numeric_array(coordinates):
            coordinates = np.asarray(coordinates, dtype=float)

            if coordinates.ndim != 3:
                raise SystemExit("Unexpected shape of coordinates.")

            if coordinates.shape[2] != 3:
                raise SystemExit("Expected each atom to have three euclidian coordinates. Got %d" % coordinates.shape[2])
        else:
            try:
                # Convert to array
                coordinates = np.asarray([np.asarray(coord, dtype=float) for coord in coordinates])

                # If all compounds have the same size, coordinates will no longer
                # be an object array, and we'll pass the array back to the function
                # to be accepted in the former if statement
                if coordinates.dtype != "object":
                    return self.set_coordinates(coordinates)
            except ValueError:
                raise SystemExit("Expected `coordinates` to be numeric array.")
            try:
                # Simple way to check that there's three coordinates for every compounds
                shapes = np.asarray([np.asarray(coord).shape for coord in coordinates], dtype=int)

                if shapes.ndim > 2:
                    raise ValueError

            except ValueError:
                raise SystemExit("Unexpected shape of coordinates.")

            if (shapes[:,1] != 3).any():
                raise SystemExit("Expected each atom to have three euclidian coordinates.")

        self._set_ncompounds(coordinates.shape[0])

        self.coordinates = coordinates

    def set_nuclear_charges(self, nuclear_charges):
        """
        Sets the nuclear_charges attribute with the given `nuclear_charges`.
        :param nuclear_charges: Array with nuclear_charges.
        :type nuclear_charges: Array
        """

        # nuclear_charges must be a numeric array or an object array where
        # each element is an array of size (?,) or (?,1), where ? can vary.
        if is_numeric_array(nuclear_charges):
            if not is_positive_integer_array(nuclear_charges):
                raise SystemExit("Expected `nuclear_charges` to be positive integer array")

            nuclear_charges = np.asarray(nuclear_charges, dtype=int)

            # Remove a third explicit dimension in necessary.
            if nuclear_charges.ndim == 3 and nuclear_charges.shape[2] == 1:
                nuclear_charges = np.squeeze(nuclear_charges, 2)

            elif nuclear_charges.ndim != 2:
                raise SystemExit("Unexpected shape of nuclear_charges.")
        else:
            try:
                # Convert to array
                nuclear_charges = np.asarray([np.asarray(char, dtype=int) for char in nuclear_charges])

                # If all compounds have the same size, coordinates will be an integer 
                # array, and we'll pass the array back to the function
                # to be accepted in the former if statement
                if nuclear_charges.dtype == "int":
                    return self.set_nuclear_charges(nuclear_charges)

            except ValueError:
                raise SystemExit("Expected `nuclear_charges` to be integer array.")
            try:
                # Simple way to check that the nuclear_charges has the expected shape.
                shapes = np.asarray([np.asarray(mol).shape for mol in nuclear_charges], dtype=int)

                if shapes.ndim > 2:
                    raise ValueError
                if shapes.shape[1] != 1:
                    # Check if the nuclear charges for compounds have shape (natoms, 1) instead of (natoms,).
                    # And if that's the case, fix it.
                    if shapes[0,1] == 1:
                        nuclear_charges = np.asarray([char.ravel() for char in nuclear_charges])
                    else:
                        raise ValueError

            except ValueError:
                raise SystemExit("Unexpected shape of nuclear_charges.")


        self._set_ncompounds(nuclear_charges.shape[0])

        self.nuclear_charges = nuclear_charges

    def remove_property(self, name):
        """
        Removes a property by name.
        """

        if not is_string(name):
            raise SystemExit("Expected the argument `name` to be a string.")

        if name in ["coordinates", "nuclear_charges"]:
            raise SystemExit("Can't remove coordinates or nuclear_charges from Data object. \
                              Try creating a new Data object instead.")

        if name in self._properties and hasattr(self, name):
            delattr(self, name)
            self._properties.pop(name)

    def __getitem__(self, i):
        """
        Required hack to use scikit-learn's cross validators.
        Basically just needs to return an object with same size
        as `i`. Also makes the attributes of the Data object
        available as a dictionary.
        """
        # Enable dictionary utility
        if i in self._properties:
            return getattr(self, i)
        return i

    def __len__(self):
        """
        Overwrite the len operator. This is a required hack to
        use scikit-learn's cross validators, but is also convenient.
        """
        return self.ncompounds

    def __eq__(self, other):
        """
        Overrides the == operator. This is a requred hack to use
        scikit-learn's cross validators, but is also convenient.
        """

        # Check if same type
        if type(self) != type(other):
            return False

        self_vars = vars(self)
        other_vars = vars(other)

        # Check that there's the same number of attributes
        if len(self_vars) != len(other_vars):
            return False

        # Check that all attributes are the same
        for key, val in self_vars.items():
            if key not in other_vars and val is not other_vars[key]:
                return False

        return True

    def __ne__(self, other):
        """
        Overrides the != operator (unnecessary in Python 3), which is needed to compliment
        the __eq__ method.
        """
        return not self.__eq__(other)

    def __repr__(self):
        """
        Output when referring to the class instance, e.g. typing x()
        in a terminal, with x being a Data class.
        """
        # Get the class name
        class_name = self.__class__.__name__
        # Get the input arguments of __init__
        arguments = inspect.getargspec(self.__init__).args[1:]

        # Construct the output string
        output = '%s(' % class_name
        for argument in arguments:
            output += argument + "=" + str(getattr(argument))

        output += ")"

        return output

    def __str__(self):
        """
        Pretty print information of data object content.
        """

        output = "Data object with %d compounds\n" % self.ncompounds
        output += "Contains the following attributes:\n"

        data = [["Name", "Property type", "Shape"]]
        data += [[key, value, str(getattr(self, key).shape)]
                for key, value in self._properties.items()]

        col_width = max(len(word) for row in data for word in row) + 2  # padding
        for i, row in enumerate(data):
            output += "\n"
            # Make the first line bold
            if i == 0:
                output += "\033[1m"
            output += "".join(word.ljust(col_width) for word in row)
            if i == 0:
                output += "\033[0m"

        return output


class XYZReader(Data):
    """
    File reader for XYZ format.
    """

    def __init__(self, filenames=None, molecular_property=None, atomic_property=None):
        """
        :param filenames: list of filenames or a string to be read by glob. e.g. 'dir/*.xyz'
        :type filenames: list or string
        :param molecular_property: Name of one or more molecular properties that occurs in the title line of the XYZ-file.
            If `molecular_property=None` no molecular properties will be read from the XYZ file. A list of names can be given,
            in which case the molecular properties are assumed to be delimited by commas or white-space. If a single name
            is given, but there is multiple molecular properties in the XYZ file, it is assumed that the property is
            multi-dimensional.
        :type molecular_property: string or list of strings
        :param atomic_property: Name of one or more atomic properties that occurs in the title line of the XYZ-file.
            If `atomic_property=None` no atomic properties will be read from the XYZ file. A list of names can be given,
            in which case the atomic properties are assumed to be delimited by commas or white-space. If a single name
            is given, but there is multiple atomic properties in the XYZ file, it is assumed that the property is
            multi-dimensional.
        :type atomic_property: string or list of strings
        """

        # Initialize parent
        super(XYZReader, self).__init__()

        if molecular_property is None or is_string(molecular_property):
            pass
        elif is_1d_array(molecular_property) and is_string(molecular_property[0]):
            if len(molecular_property) == 1:
                molecular_property = molecular_property[0]
        else:
            raise SystemExit("Expected `molecular_property` to be a string or array of strings. Got %s" % str(molecular_property))

        if atomic_property is None or is_string(atomic_property):
            pass
        elif is_1d_array(atomic_property) and is_string(atomic_property[0]):
            if len(atomic_property) == 1:
                atomic_property = atomic_property[0]
        else:
            raise SystemExit("Expected `atomic_property` to be a string or array of strings. Got %s" % str(atomic_property))

        self.filenames = filenames
        if is_string(self.filenames):
            self.filenames = sorted(glob.glob(self.filenames))
        if is_1d_array(self.filenames) and is_string(self.filenames[0]):
            self._parse_xyz_files(self.filenames, molecular_property, atomic_property)
        else:
            raise SystemExit("`filenames` must be a string to be read by glob or a list of strings")

    def get_filenames(self):
        """
        Returns a list of filenames in the order they were parsed.
        """
        return self.filenames

    def _parse_xyz_files(self, filenames, molecular_property_name, atomic_property_name):
        """
        Parse a list of xyz files and the molecular and atomic properties.
        """
        def parse_properties(tokens, n_properties):
            n_tokens = len(tokens)

            if n_tokens == 0:
                raise SystemExit("Expected %d properties, but none was present in XYZ file" % n_properties)
            elif n_tokens == 1:
                if n_properties == 1:
                    prop = [tokens[0]]
                else:
                    raise SystemExit("Read 1 property from XYZ file, but expected %d" % n_properties)
            else:
                if n_properties == 1 or n_properties == n_tokens:
                    prop = tokens
                else:
                    raise SystemExit("Expected %d properties, but %d was found in XYZ file" % (n_properties, n_tokens))

            # Catch non-numeric properties
            if is_numeric_1d_array(prop):
                if is_integer_array(prop):
                    prop = np.asarray(prop, dtype=int)
                else:
                    prop = np.asarray(prop, dtype=float)
            else:
                raise SystemExit("Non-numeric property in XYZ file")

            return prop

        def get_property_size(prop):
            if prop is None:
                n = 0
            if is_string(prop):
                n = 1
            else:
                n = len(prop)
            return n

        coordinates = np.empty(self.ncompounds, dtype=object)
        nuclear_charges = np.empty(self.ncompounds, dtype=object)

        # Get expected property sizes
        n_molprop = get_property_size(molecular_property_name)
        n_atprop = get_property_size(atomic_property_name)

        if n_molprop > 0:
            molecular_properties = []
        if n_atprop > 0:
            atomic_properties = []

        # Report if there's any error in reading the xyz files.
        # This is a lazy approach, but the XYZ reader might be replaced
        # with ASE at some point.
        try:
            for i, filename in enumerate(filenames):
                with open(filename, "r") as f:
                    lines = f.readlines()

                natoms = int(lines[0])

                #Handle molecular properties
                tokens = lines[1].split()
                if n_molprop > 0:
                    molecular_properties.append(parse_properties(tokens, n_molprop))

                nuclear_charges[i] = np.empty(natoms, dtype=int)
                coordinates[i] = np.empty((natoms, 3), dtype=float)

                if n_atprop > 0:
                    # Atomic properties for this compound
                    atomic = []
                for j, line in enumerate(lines[2:natoms+2]):
                    tokens = line.split()

                    if len(tokens) < 4:
                        raise SystemExit("Error in parsing coordinates in XYZ file")

                    #Handle atomic properties
                    if n_atprop > 0:
                        atomic.append(parse_properties(tokens[4:], n_atprop))

                    nuclear_charges[i][j] = NUCLEAR_CHARGE[tokens[0]]
                    coordinates[i][j] = np.asarray(tokens[1:4], dtype=float)

                if n_atprop > 0:
                    atomic = np.asarray(atomic)
                    if atomic.dtype == "object":
                        raise SystemExit("Different number of atomic properties found in %s" % filename)
                    atomic_properties.append(prop)
                    if i > 0:
                        shape_this = atomic_properties[i].shape
                        shape_last = atomic_properties[i-1].shape
                        # Make sure that there's not different number of atomic properties
                        # in the XYZ files
                        if shape_this[1:] != shape_last[1:]:
                            raise SystemExit("Different number of atomic properties in %s and %s" % (filename, filenames[i-1]))
        except:
            raise SystemExit("Error in reading %s" % filename)

        # Set coordinates and nuclear_charges
        self.set_coordinates(coordinates)
        self.set_nuclear_charges(nuclear_charges)

        # Handle setting properties
        if n_molprop > 0:
            molecular_properties = np.asarray(molecular_properties)
            if molecular_properties.dtype == "object":
                raise SystemExit("Different number of molecular properties found in XYZ files")
            if n_molprop == 1:
                self.add_molecular_property(molecular_properties, molecular_property_name)
            else:
                for name, prop in zip(molecular_property_name, molecular_properties.T):
                    self.add_molecular_property(prop, name)

        if n_atprop > 0:
            atomic_properties = np.asarray(atomic_properties)
            # If there's different sized compounds, we have to do list comprehensions
            if n_atprop == 1:
                self.add_atomic_property(atomic_properties, atomic_property_name)
            else:
                for i in range(n_atprop):
                    name = atomic_property_name[i]
                    if atomic_properties.dtype == "object":
                        prop = np.asarray([mol[:,i] for mol in atomic_properties])
                    else:
                        prop = atomic_properties[:,:,i]

                    self.add_atomic_property(prop, name)
