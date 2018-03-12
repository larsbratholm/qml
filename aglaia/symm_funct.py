"""
This module contains an implementation of the the symmetry functions used in the Parkhill paper https://arxiv.org/pdf/1711.06385.pdf.
This implementation is different.
Note: it is all in single precision.
"""

import tensorflow as tf
import numpy as np

def acsf_rad(xyzs, Zs, radial_cutoff, radial_rs, eta):
    """
    This does the radial part of the symmetry function (G2 function in Behler's papers). It doesn't distinguish for
    different pairs of atoms

    :param xyzs: tf tensor of shape (n_samples, n_atoms, 3) contaning the coordinates of each atom in each data sample
    :param Zs: tf tensor of shape (n_samples, n_atoms) containing the atomic number of each atom in each data sample
    :param radial_cutoff: scalar tensor
    :param radial_rs: tf tensor of shape (n_rs,) with the R_s values
    :param eta: tf scalar

    :return: tf tensor of shape (n_samples, n_atoms, n_rs)
    """

    # Calculating the distance matrix between the atoms of each sample
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    dist_tensor = tf.cast(tf.norm(dxyzs, axis=3), dtype=tf.float32)  # (n_samples, n_atoms, n_atoms)

    # Indices of terms that need to be zero (diagonal elements)
    n_atoms = Zs.get_shape().as_list()[1]
    n_samples = Zs.get_shape().as_list()[0]

    zarray = np.zeros((n_samples, n_atoms, n_atoms))

    for i in range(n_atoms):
                    zarray[:, i, i] = 1

    where_eq_idx = tf.convert_to_tensor(zarray, dtype=tf.bool)

    # Calculating the exponential term
    expanded_rs = tf.expand_dims(tf.expand_dims(tf.expand_dims(radial_rs, axis=0), axis=0), axis=0) # (1, 1, 1, n_rs)
    expanded_dist = tf.expand_dims(dist_tensor, axis=-1) # (n_samples, n_atoms, n_atoms, 1)
    exponent = - eta * tf.square(tf.subtract(expanded_dist, expanded_rs))
    exp_term = tf.exp(exponent) # (n_samples, n_atoms, n_atoms, n_rs)

    # Calculating the fc terms
    # Finding where the distances are less than the cutoff
    where_less_cutoff = tf.less(dist_tensor, radial_cutoff)
    # Calculating all of the fc function terms
    fc = 0.5 * (tf.cos(3.14159265359 * dist_tensor / radial_cutoff) + 1.0)
    # Setting to zero the terms where the distance is larger than the cutoff
    zeros = tf.zeros(tf.shape(dist_tensor))
    cut_off_fc = tf.where(where_less_cutoff, fc, zeros)  # (n_samples, n_atoms, n_atoms)
    # Cleaning up diagonal terms
    clean_fc_term = tf.where(where_eq_idx, zeros, cut_off_fc)

    # Multiplying exponential and fc terms
    expanded_fc = tf.expand_dims(clean_fc_term, axis=-1) # (n_samples, n_atoms, n_atoms, 1)
    presum_term = tf.multiply(expanded_fc, exp_term) # (n_samples, n_atoms, n_atoms, n_rs)

    return presum_term

def acsf_ang(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    """
    This does the angular part of the symmetry function as mentioned here: https://arxiv.org/pdf/1711.06385.pdf
    At present it does not make different bins depending on the atom type. It also works for systems where all the
    samples are the same molecule.

    :param xyzs: tf tensor of shape (n_samples, n_atoms, 3) contaning the coordinates of each atom in each data sample
    :param Zs: tf tensor of shape (n_samples, n_atoms) containing the atomic number of each atom in each data sample
    :param angular_cutoff: scalar tensor
    :param angular_rs: tf tensor of shape (n_ang_rs,) with the equivalent of the R_s values from the G2
    :param theta_s: tf tensor of shape (n_thetas,)
    :param zeta: tf tensor of shape (1,)
    :param eta: tf tensor of shape (1,)
    :return: tf tensor of shape (n_samples, n_atoms, n_ang_rs * n_thetas)
    """

    # Finding the R_ij + R_ik term
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    dist_tensor = tf.cast(tf.norm(dxyzs, axis=3), dtype=tf.float32)  # (n_samples, n_atoms, n_atoms)

    # This is the tensor where element sum_dist_tensor[0,1,2,3] is the R_12 + R_13 in the 0th data sample
    sum_dist_tensor = tf.expand_dims(dist_tensor, axis=3) + tf.expand_dims(dist_tensor,
                                                                           axis=2)  # (n_samples, n_atoms, n_atoms, n_atoms)

    # Problem with the above tensor: we still have the R_ii + R_ik distances which are non zero and could be summed
    # These need to be set to zero
    n_atoms = Zs.get_shape().as_list()[1]
    n_samples = Zs.get_shape().as_list()[0]

    # Make an array of zero and turn all the elements where two indices are the same to 1
    zarray = np.zeros((n_samples, n_atoms, n_atoms, n_atoms))

    # Find a vectorised way of doing this
    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(n_atoms):
                if i == j or i == k or j == k:
                    zarray[:, i, j, k] = 1

    # Make a bool tensor of the indices
    where_eq_idx = tf.convert_to_tensor(zarray, dtype=tf.bool)

    # For all the elements that are true in where_eq_idx, turn the elements of sum_dist_tensor to zero
    zeros_1 = tf.zeros(tf.shape(sum_dist_tensor))
    clean_sum_dist = tf.where(where_eq_idx, zeros_1, sum_dist_tensor)  # (n_samples,  n_atoms, n_atoms, n_atoms)
    # (Maybe there is no need for this last step, because the terms will be zeroed once they are multiplied by the clean_fc_term)

    # Now finding the fc terms
    # 1. Find where Rij and Rik are < cutoff
    where_less_cutoff = tf.less(dist_tensor, angular_cutoff)
    # 2. Calculate the fc on the Rij and Rik tensors
    fc_1 = 0.5 * (tf.cos(3.14159265359 * dist_tensor / angular_cutoff) + 1.0)
    # 3. Apply the mask calculated in 1.  to zero the values for where the distances are > than the cutoff
    zeros_2 = tf.zeros(tf.shape(dist_tensor))
    cut_off_fc = tf.where(where_less_cutoff, fc_1, zeros_2)  # (n_samples, n_atoms, n_atoms)
    # 4. Multiply the two tensors elementwise
    fc_term = tf.multiply(tf.expand_dims(cut_off_fc, axis=3),
                          tf.expand_dims(cut_off_fc, axis=2))  # (n_samples,  n_atoms, n_atoms, n_atoms)
    # 5. Cleaning up the terms that should be zero because there are equal indices
    clean_fc_term = tf.where(where_eq_idx, zeros_1, fc_term)

    # Now finding the theta_ijk term
    # Finding the tensor with the vector distances between all the atoms
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    # Doing the dot products of all the possible vectors
    dots_dxyzs = tf.cast(tf.reduce_sum(tf.multiply(tf.expand_dims(dxyzs, axis=3), tf.expand_dims(dxyzs, axis=2)),
                               axis=4), dtype=tf.float32)  # (n_samples,  n_atoms, n_atoms, n_atoms)
    # Doing the products of the magnitudes
    dist_prod = tf.multiply(tf.expand_dims(dist_tensor, axis=3),
                            tf.expand_dims(dist_tensor, axis=2))  # (n_samples,  n_atoms, n_atoms, n_atoms)
    # Dividing the dot products by the magnitudes to obtain cos theta
    cos_theta = tf.divide(dots_dxyzs, dist_prod)
    # Applying arc cos to find the theta value
    theta = tf.acos(cos_theta)  # (n_samples,  n_atoms, n_atoms, n_atoms)
    # Removing the NaNs created by dividing by zero AND setting to zero the elements from angles created by R_ij x R_ij
    clean_theta = tf.where(where_eq_idx, zeros_1, theta)

    # Finding the (0.5 * clean_sum_dist - R_s) term
    # Augmenting the dims of angular_rs
    expanded_rs = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(angular_rs, axis=0), axis=0), axis=0),
                                 axis=0)  # (1, 1, 1, 1, n_rs)
    # Augmenting the dim of clean_sum_dist *0.5
    expanded_sum = tf.expand_dims(clean_sum_dist * 0.5, axis=-1)
    # Combining them
    brac_term = tf.subtract(expanded_sum, expanded_rs)
    # Finally making the exponential term
    exponent = - eta * tf.square(brac_term)
    exp_term = tf.exp(exponent)  # (n_samples,  n_atoms, n_atoms, n_atoms, n_rs)

    # Finding the cos(theta - theta_s) term
    # Augmenting the dimensions of theta_s
    expanded_theta_s = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(theta_s, axis=0), axis=0), axis=0),
                                      axis=0)
    # Augmenting the dimensions of theta
    expanded_theta = tf.expand_dims(clean_theta, axis=-1)
    # Subtracting them and do the cos
    cos_theta_term = tf.cos(
        tf.subtract(expanded_theta, expanded_theta_s))  # (n_samples,  n_atoms, n_atoms, n_atoms, n_theta_s)
    # Make the whole cos term  of the sum
    cos_term = tf.pow(tf.add(tf.ones(tf.shape(cos_theta_term)), cos_theta_term),
                      zeta)  # (n_samples,  n_atoms, n_atoms, n_atoms, n_theta_s)

    # Final product of terms inside the sum time by 2^(1-zeta)
    expanded_fc = tf.expand_dims(tf.expand_dims(clean_fc_term, axis=-1), axis=-1)
    expanded_cos = tf.expand_dims(cos_term, axis=-2)
    expanded_exp = tf.expand_dims(exp_term, axis=-1)

    const = tf.pow(2.0, (1.0 - zeta))
    prod_of_terms = const * tf.multiply(tf.multiply(expanded_cos, expanded_exp),
                                        expanded_fc)  # (n_samples,  n_atoms, n_atoms, n_atoms, n_rs, n_theta_s)

    # Reshaping to shape (n_samples,  n_atoms, n_atoms, n_atoms, n_rs*n_theta_s)
    presum_term = tf.reshape(prod_of_terms,
                             [n_samples, n_atoms, n_atoms, n_atoms, tf.shape(theta_s)[0] * tf.shape(angular_rs)[0]])

    return presum_term

def sum_rad(pre_sum, Zs, elements_list, radial_rs):
    """
    Sum of the terms in the radial part of the symmetry function. The terms corresponding to the same neighbour identity
    are summed together.

    :param pre_sum: tf tensor of shape (n_samples, n_atoms, n_atoms, n_rs)
    :param Zs: tf tensor of shape (n_samples, n_atoms)
    :param elements_list: np.array of shape (n_elements,)
    :param radial_rs: tf tensor of shape (n_rad_rs,)
    :return: tf tensor of shape (n_samples, n_atoms, n_rad_rd * n_elements)
    """
    n_atoms = Zs.get_shape().as_list()[1]
    n_samples = Zs.get_shape().as_list()[0]
    n_elements = len(elements_list)
    n_rs = radial_rs.get_shape().as_list()[0]

    ## Making a matrix of all the possible neighbouring atoms
    # No need to clean up diagonal elements because they are already set to zero in the presum term
    neighb_atoms = tf.tile(tf.expand_dims(tf.expand_dims(Zs, axis=1), axis=-1),
                           multiples=[1, n_atoms, 1, n_rs])  # (n_samples, n_atoms, n_atoms, n_rs)
    zeros = tf.zeros(tf.shape(pre_sum), dtype=tf.float32)


    # Looping over all the possible elements in the system and extracting the relevant terms from the pre_sum term
    pre_sum_terms = []

    for i in range(n_elements):
        element = tf.constant(elements_list[i], dtype=tf.int32)
        expanded_element = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(element, axis=0), axis=0), axis=0), axis=0),
            multiples=[n_samples, n_atoms, n_atoms, n_rs])
        equal_elements =  tf.equal(expanded_element, neighb_atoms)
        slice_presum = tf.where(equal_elements, pre_sum, zeros)
        pre_sum_terms.append(slice_presum)

    # Concatenating the extracted terms and summing the ones corresponding to the same atom types.
    rad_acsf_presum = tf.concat(pre_sum_terms, axis=-1)
    final_term = tf.reduce_sum(rad_acsf_presum, axis=[2])

    return final_term

def sum_ang(pre_sumterm, Zs, element_pairs_list, angular_rs, theta_s):
    """
    This function does the sum of the terms in the radial part of the symmetry function. Three body interactions where
    the two neighbours are the same elements are summed together.

    :param pre_sumterm: tf tensor of shape (n_samples, n_atoms, n_ang_rs * n_thetas)
    :param Zs: tf tensor of shape (n_samples, n_atoms)
    :param element_pairs_list: np array of shape (n_elementpairs, 2)
    :param angular_rs: tf tensor of shape (n_ang_rs,)
    :param theta_s: tf tensor of shape (n_thetas,)
    :return: tf tensor of shape (n_samples, n_atoms, n_ang_rs * n_thetas * n_elementpairs)
    """

    n_atoms = Zs.get_shape().as_list()[1]
    n_samples = Zs.get_shape().as_list()[0]
    n_pairs = len(element_pairs_list)
    n_rs = angular_rs.get_shape().as_list()[0]
    n_thetas = theta_s.get_shape().as_list()[0]

    # Making the pair matrix
    Zs_exp_1 = tf.expand_dims(tf.tile(tf.expand_dims(Zs, axis=1), multiples=[1, n_atoms, 1]), axis=-1)
    Zs_exp_2 = tf.expand_dims(tf.tile(tf.expand_dims(Zs, axis=-1), multiples=[1, 1, n_atoms]), axis=-1)
    neighb_pairs = tf.concat([Zs_exp_1, Zs_exp_2], axis=-1)  # (n_samples, n_atoms, n_atoms, 2)

    # Cleaning up diagonal elements
    zarray = np.zeros((n_samples, n_atoms, n_atoms, 2))

    for i in range(n_atoms):
        zarray[:, i, i, :] = 1

    where_eq_idx = tf.convert_to_tensor(zarray, dtype=tf.bool)
    zeros = tf.zeros(tf.shape(neighb_pairs), dtype=tf.int32)
    clean_pairs = tf.where(where_eq_idx, zeros, neighb_pairs)

    # Sorting the pairs in descending order so that for example pair [7, 1] is the same as [1, 7]
    sorted_pairs, _ = tf.nn.top_k(clean_pairs, k=2, sorted=True)  # (n_samples, n_atoms, n_atoms, 2)

    # Preparing to clean the sorted pairs from where there will be self interactions in the three-body-terms
    oarray = np.ones((n_samples, n_atoms, n_atoms, n_atoms))

    # Find a vectorised way of doing this
    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(n_atoms):
                if i == j or i == k or j == k:
                    oarray[:, i, j, k] = 0

    where_self_int = tf.convert_to_tensor(oarray, dtype=tf.bool)
    exp_self_int = tf.expand_dims(where_self_int, axis=-1)  # (n_samples, n_atoms, n_atoms, n_atoms, 1)

    zeros_large = tf.zeros(tf.shape(pre_sumterm), dtype=tf.float32)
    presum_terms = []

    for i in range(n_pairs):
        # Making a tensor where all the elements are the pair under consideration
        pair = tf.constant(element_pairs_list[i], dtype=tf.int32)
        expanded_pair = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(pair, axis=0), axis=0), axis=0),
            multiples=[n_samples, n_atoms, n_atoms, 1])  # (n_samples, n_atoms, n_atoms, 2)
        # Comparing which neighbour pairs correspond to the pair under consideration
        equal_pair_mix = tf.equal(expanded_pair, sorted_pairs)
        equal_pair_split1, equal_pair_split2 = tf.split(equal_pair_mix, 2, axis=-1)
        equal_pair = tf.tile(tf.expand_dims(tf.logical_and(equal_pair_split1, equal_pair_split2), axis=[1]),
                             multiples=[1, n_atoms, 1, 1, 1])  # (n_samples, n_atoms, n_atoms, n_atoms, 1)
        # Removing the pairs where the same atom is present more than once
        int_to_keep = tf.logical_and(equal_pair, exp_self_int)
        exp_int_to_keep = tf.tile(int_to_keep, multiples=[1, 1, 1, 1, n_rs * n_thetas])
        # Extracting the terms that correspond to the pair under consideration
        slice_presum = tf.where(exp_int_to_keep, pre_sumterm, zeros_large)
        presum_terms.append(slice_presum)

    # Concatenating all of the terms corresponding to different pair neighbours
    angular_acsf_presum = tf.concat(presum_terms, axis=-1)
    # Summing over neighbouring pairs
    final_term = 0.5 * tf.reduce_sum(angular_acsf_presum, axis=[2, 3])

    return final_term

def generate_parkhill_acsf(xyzs, Zs, elements, element_pairs, radial_cutoff=10.0, angular_cutoff=10.0,
                           radial_rs=None, angular_rs=None, theta_s=None, zeta=3.0, eta=2.0):
    """
    This function generates the atom centred symmetry function as used in the Tensormol paper. Currently only tested for
    single systems with many conformations. It requires the coordinates of all the atoms in each data sample, the atomic
    charges for each atom (in the same order as the xyz), the overall elements and overall element pairs. Then it
    requires the parameters for the ACSF that are used in the Tensormol paper: https://arxiv.org/pdf/1711.06385.pdf

    :param xyzs: np.array of shape (n_samples, n_atoms, 3)
    :param Zs: np.array of shape (n_samples, n_atoms)
    :param elements: np.array of shape (n_elements,)
    :param element_pairs: np.array of shape (n_elementpairs, 2)
    :param radial_cutoff: scalar float
    :param angular_cutoff: scalar float
    :param radial_rs: np.array of shape (n_rad_rs,)
    :param angular_rs: np.array of shape (n_ang_rs,)
    :param theta_s: np.array of shape (n_thetas,)
    :param zeta: scalar float
    :param eta: scalar float
    :return: a tf tensor of shape (n_samples, n_atoms, n_rad_rs * n_elements + n_ang_rs * n_thetas * n_elementpairs)
    """

    # Checking if the parameters that should be lists have been left to the default values
    if type(radial_rs) == type(None):
        radial_rs = [0.0, 0.1, 0.2]
    if type(angular_rs) == type(None):
        angular_rs = [0.0, 0.1, 0.2]
    if type(theta_s) == type(None):
        theta_s = [3.0, 2.0]

    # Turning the quantities into tensors
    Zs_tf = tf.constant(Zs, dtype=tf.int32)
    xyzs_tf = tf.constant(xyzs, dtype=tf.float32)

    rad_cutoff = tf.constant(radial_cutoff, dtype=tf.float32)
    ang_cutoff = tf.constant(angular_cutoff, dtype=tf.float32)
    rad_rs = tf.constant(radial_rs, dtype=tf.float32)
    ang_rs = tf.constant(angular_rs, dtype=tf.float32)
    theta_s = tf.constant(theta_s, dtype=tf.float32)
    zeta_tf = tf.constant(zeta, dtype=tf.float32)
    eta_tf = tf.constant(eta, dtype=tf.float32)

    ##  Calculating the radial part of the symmetry function
    # First obtaining all the terms in the sum
    pre_sum_rad = acsf_rad(xyzs_tf, Zs_tf, rad_cutoff, rad_rs, eta_tf)  # (n_samples, n_atoms, n_atoms, n_rad_rs)
    # Then summing based on the identity of the atoms interacting
    rad_term = sum_rad(pre_sum_rad, Zs_tf, elements, rad_rs) # (n_samples, n_atoms, n_rad_rs*n_elements)

    ## Calculating the angular part of the symmetry function
    # First obtaining all the terms in the sum
    pre_sum_ang = acsf_ang(xyzs_tf, Zs_tf, ang_cutoff, ang_rs, theta_s, zeta_tf, eta_tf) # (n_samples, n_atoms, n_atoms, n_atoms, n_thetas * n_ang_rs)
    # Then doing the sum based on the neighbrouing pair identity
    ang_term = sum_ang(pre_sum_ang, Zs_tf, element_pairs, ang_rs, theta_s) # (n_samples, n_atoms, n_thetas * n_ang_rs*n_elementpairs)

    acsf = tf.concat([rad_term, ang_term], axis=-1) # (n_samples, n_atoms, n_rad_rs*n_elements + n_thetas * n_ang_rs*n_elementpairs)

    return acsf
