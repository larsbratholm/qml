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

from ..utils import NUCLEAR_CHARGE

#TODO:
# Finish XYZ file reader
# Make properties available as dictionaries
# Set property method, that includes name and molecular/atomic/pairs
# __str__ and __repr__ methods
# getattr_subset method that gets x[self._indices]

class Data(object):
    """
    Scikit-learn compatible Data class. Different file readers / Dataset objects 
    should inherit from this to maintain compatibility.

    """

    def __init__(self):

        self._set_ncompounds(0)
        self.coordinates = None
        self.nuclear_charges = None
        self.natoms = None
        self.shape = None
        self._indices = None
        self.properties = {'coordinates': self.coordinates,
                           'nuclear_charges': self.nuclear_charges}

        if isinstance(filenames, str):
            filenames = sorted(glob.glob(filenames))
        if isinstance(filenames, list):
            self._parse_xyz_files(filenames)
        # Overwritten in various parts of a standard prediction pipeline
        # so don't use these within the class
        #self._has_transformed_labels
        #self._representations
        #self._kernel
        #self._representation_type
        #self._representation_short_name
        #self._representation_cutoff
        #self._representation_alchemy

    def _set_ncompounds(self, n):
        """
        Updates the number of compounds stored in the class
        """
        self.ncompounds = n
        # Hack for sklearn CV
        self.shape = (n,)
        self._indices = np.arange(n)

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

    def __getitem__(self, i):
        """
        Required hack to use scikit-learn's cross validators.
        Basically just needs to return an object with same size
        as `i`
        """
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

        output = "Data object with %d compounds" % self.ncompounds
        output += "\n\n"
        output += "Contains the following attributes: \n\n"

        col_width = max(len(word) for row in data for word in row) + 2  # padding
        #"Name", "Property type", "Shape"
        for row in data:
            print "".join(word.ljust(col_width) for word in row)


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
            in which case the molecular properties are assumed to be delimited by commas or whitespace.
        :type molecular_property: string or list of strings
        """

    def _parse_xyz_files(self, filenames):
        """
        Parse a list of xyz files.
        """

        self._set_ncompounds(len(filenames))
        self.coordinates = np.empty(self.ncompounds, dtype=object)
        self.nuclear_charges = np.empty(self.ncompounds, dtype=object)
        self.natoms = np.empty(self.ncompounds, dtype = int)

        for i, filename in enumerate(filenames):
            with open(filename, "r") as f:
                lines = f.readlines()

            natoms = int(lines[0])
            self.natoms[i] = natoms
            self.nuclear_charges[i] = np.empty(natoms, dtype=int)
            self.coordinates[i] = np.empty((natoms, 3), dtype=float)

            for j, line in enumerate(lines[2:natoms+2]):
                tokens = line.split()

                if len(tokens) < 4:
                    break

                self.nuclear_charges[i][j] = NUCLEAR_CHARGE[tokens[0]]
                self.coordinates[i][j] = np.asarray(tokens[1:4], dtype=float)

        # Try to convert dtype to int/float in cases where you have the
        # same molecule, just different conformers

        try:
            self.nuclear_charges = np.asarray([self.nuclear_charges[i] for i in range(self.ncompounds)], 
                    dtype=int)
            self.coordinates = np.asarray([self.coordinates[i] for i in range(self.ncompounds)],
                    dtype=float)
        except ValueError:
            pass



