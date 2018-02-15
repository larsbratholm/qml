import tensorflow as tf
import numpy as np

def symm_func_g2(xyzs, Zs, radial_cutoff, radial_rs, eta):
    """
    This does the radial part of the symmetry function (G2 function in Behler's papers). It calculates each term for the
    sum, but it doesn't do the sum.

    :param xyzs: tf tensor of shape (n_samples, n_atoms, 3) contaning the coordinates of each atom in each data sample
    :param Zs: tf tensor of shape (n_samples, n_atoms) containing the atomic number of each atom in each data sample
    :param radial_cutoff: scalar tensor
    :param radial_rs: tf tensor of shape (n_rs,) with the R_s values
    :param eta: tf tensor of shape (n_eta,) with the eta values

    :return: tf tensor of shape (n_samples * n_atoms * n_neighbours, n_rs * n_eta)
    """

    # Calculating the distance matrix between the atoms of each sample
    dxyzs = tf.expand_dims(xyzs, axis=2) - tf.expand_dims(xyzs, axis=1)
    dist_tensor = tf.norm(dxyzs + 1.e-16, axis=3)  # (n_samples, n_atoms, n_atoms)

    padding_mask = tf.not_equal(Zs, 0)  # shape (n_samples, n_atoms)
    expanded_padding_1 = tf.expand_dims(padding_mask, axis=1)  # (n_samples, 1, n_atoms)
    expanded_padding_2 = tf.expand_dims(padding_mask, axis=-1)  # (n_samples, n_atoms, 1)
    under_cutoff = tf.less(dist_tensor,
                           radial_cutoff)  # Only for distances < cut-off this is true, same shape as dist_tensor
    # If there is an atom AND the distance is < cut-off, then mask2 element is TRUE (done one dimension at a time)
    mask1 = tf.logical_and(under_cutoff, expanded_padding_1)  # (n_samples, n_atoms, n_atoms)
    mask2 = tf.logical_and(mask1, expanded_padding_2)  # (n_samples, n_atoms, n_atoms)

    # All the indices of the atoms-pairs that have distances inside the cut-off
    pair_indices = tf.where(mask2)  # (n, 3)
    # Removing diagonal elements
    identity_mask = tf.where(tf.not_equal(pair_indices[:, 1], pair_indices[:, 2]))
    pair_indices = tf.cast(tf.squeeze(tf.gather(pair_indices, identity_mask)), tf.int32)  # (n_pairs, 3)

    # Making a tensor with the distances of the atom pairs corresponding to the indices in pair_indices
    pair_distances = tf.gather_nd(dist_tensor, pair_indices) # (n_pairs,)

    # Gathering the atomic numbers of the atoms in the atom-pairs
    pair_elements = tf.stack([tf.gather_nd(Zs, pair_indices[:, 0:2]), tf.gather_nd(Zs, pair_indices[:, 0:3:2])],
                             axis=-1)

    # Calculating the argument of the sum in the G2 function (without the cut-off part) - shape (n_pairs, 1)
    gaussian_factor = tf.exp(
        -eta * tf.square(tf.expand_dims(pair_distances, axis=-1) - tf.expand_dims(radial_rs, axis=0)))

    # Calculating the cutoff term of the sum
    cutoff_factor = tf.expand_dims(0.5 * (tf.cos(3.14159265359 * pair_distances / radial_cutoff) + 1.0), axis=-1)

    # Actual symmetry function
    radial_embedding = gaussian_factor * cutoff_factor

    return radial_embedding

def symm_func_salpha_ang(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta):
    """
    This does the angular part of the symmetry function as mentioned here: https://arxiv.org/pdf/1711.06385.pdf

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
    dist_tensor = tf.norm(dxyzs, axis=3)  # (n_samples, n_atoms, n_atoms)

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
    dots_dxyzs = tf.reduce_sum(tf.multiply(tf.expand_dims(dxyzs, axis=3), tf.expand_dims(dxyzs, axis=2)),
                               axis=4)  # (n_samples,  n_atoms, n_atoms, n_atoms)
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

    # Summing the dimensions to get a term of shape (n_samples,  n_atoms, n_rs*n_theta_s)
    final_term = tf.reduce_sum(presum_term, axis=[2, 3])

    return final_term