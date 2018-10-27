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

"""
Osprey wrappers for neural network classes
"""

from __future__ import print_function
from .aglaia import ARMP_G
import numpy as np
import tensorflow as tf

class ARMP_G_Wrapper(ARMP_G):

    def __init__(self, hl1=5, hl2=0, hl3=0, hl4=0, l1_reg=0.0, l2_reg=0.0001, batch_size='auto', learning_rate=0.001,
                 iterations=500, scoring_function='negmae', forces_score_weight=0.0, phi=0.0,
                 acsf_cutoff=5.0, acsf_nbasis=5, acsf_precision=2, coordinates=None, nuclear_charges=None,
                 energies=None, forces=None):


        self.hl1 = hl1
        self.hl2 = hl2
        self.hl3 = hl3
        self.hl4 = hl4
        self.acsf_cutoff = acsf_cutoff
        self.acsf_precision = acsf_precision
        self.acsf_nbasis = acsf_nbasis
        self.coordinates = coordinates
        self.nuclear_charges = nuclear_charges
        self.energies = energies
        self.forces = forces


        super(ARMP_G_Wrapper, self).__init__(l1_reg=l1_reg, l2_reg=l2_reg,
                batch_size=batch_size, learning_rate=learning_rate, iterations=iterations,
                scoring_function=scoring_function, forces_score_weight=forces_score_weight,
                phi=phi)

    def fit(self, x, y=None):
        """
        Fit the network

        :param x: indices
        :type x: numpy array of shape (n_samples, n_features)
        :param y: dummy for sklearn compatibility
        :type y: None
        """

        # Sets activation_function, tf_dtypes and hidden_layers at fit time for sklearn compatibility
        self._update_activation_function()
        self._update_tf_dtype()
        self.hidden_layer_sizes = np.asarray([n for n in 
                2**np.asarray([self.hl1, self.hl2, self.hl3, self.hl4], dtype=int) if n > 1])
        eta = 4 * np.log(self.acsf_precision) * ((self.acsf_nbasis-1)/(self.acsf_cutoff))**2
        zeta = - np.log(self.acsf_precision) / np.log(np.cos(np.pi / (4 * (self.acsf_nbasis - 1)))**2)

        self.acsf_parameters = {'rcut': self.acsf_cutoff, 'acut': self.acsf_cutoff, 'nRs2': self.acsf_nbasis,
                'nRs3': self.acsf_nbasis, 'nTs': self.acsf_nbasis, 'zeta': zeta, 'eta': eta}

        # Osprey converts int to float, so revert that
        idx = x.ravel().astype(int)
        tf.reset_default_graph()

        return self._fit(self.coordinates[idx], self.energies[idx],
                self.nuclear_charges[idx], self.forces[idx])

    def score(self, x, y=None):
        # Osprey converts int to float, so revert that
        idx = x.ravel().astype(int)
        return self._score(self.coordinates[idx], self.energies[idx],
                self.nuclear_charges[idx], self.forces[idx])



