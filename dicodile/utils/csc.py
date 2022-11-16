"""Helper functions for Convolutional Sparse Coding.

Author : tommoral <thomas.moreau@inria.fr>
"""

import numpy as np
from scipy import signal

from .shape_helpers import get_full_support, get_valid_support


def compute_ztz(z, atom_support, padding_support=None):
    """
    ztz.shape = n_atoms, n_atoms, 2 * atom_support - 1
    z.shape = n_atoms, n_times - n_times_atom + 1)
    """
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, *_ = z.shape
    ztz_shape = (n_atoms, n_atoms) + tuple(2 * np.array(atom_support) - 1)

    if padding_support is None:
        padding_support = [(size_atom_ax - 1, size_atom_ax - 1)
                           for size_atom_ax in atom_support]

    padding_shape = np.asarray([(0, 0)] + padding_support, dtype='i')
    inner_slice = (Ellipsis,) + tuple([
        slice(size_atom_ax - 1, - size_atom_ax + 1)
        for size_atom_ax in atom_support])

    z_pad = np.pad(z, padding_shape, mode='constant')
    z = z_pad[inner_slice]

    # Choose between sparse and fft
    z_nnz = z.nonzero()
    ratio_nnz = len(z_nnz[0]) / z.size
    if ratio_nnz < .05:
        ztz = np.zeros(ztz_shape)
        for k0, *pt in zip(*z_nnz):
            z_pad_slice = tuple([slice(None)] + [
                slice(v, v + 2 * size_ax - 1)
                for v, size_ax in zip(pt, atom_support)])
            ztz[k0] += z[(k0, *pt)] * z_pad[z_pad_slice]
    else:
        # compute the cross correlation between z and z_pad
        z_pad_reverse = np.flip(z_pad, axis=tuple(range(1, z.ndim)))
        ztz = np.array([[signal.fftconvolve(z_pad_k0, z_k, mode='valid')
                         for z_k in z]
                        for z_pad_k0 in z_pad_reverse])
    assert ztz.shape == ztz_shape, (ztz.shape, ztz_shape)
    return ztz


def compute_ztX(z, X):
    """
    z.shape = n_atoms, n_times - n_times_atom + 1)
    X.shape = n_channels, n_times
    ztX.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, *valid_support = z.shape
    n_channels, *sig_support = X.shape
    atom_support = get_valid_support(sig_support, valid_support)

    ztX = np.zeros((n_atoms, n_channels, *atom_support))
    for k, *pt in zip(*z.nonzero()):
        pt = tuple(pt)
        X_slice = (Ellipsis,) + tuple([
            slice(v, v + size_atom_ax)
            for v, size_atom_ax in zip(pt, atom_support)
        ])
        ztX[k] += z[k][pt] * X[X_slice]

    return ztX


def soft_thresholding(x, mu, positive=False):
    """Soft-thresholding point-wise operator

    Parameters
    ----------
    x : ndarray
        Variable on which the soft-thresholding is applied.
    mu : float
        Threshold of the operator
    positive : boolean
        If set to True, apply the soft-thresholding with positivity constraint.
    """
    if positive:
        return np.maximum(x - mu, 0)

    return np.sign(x) * np.maximum(abs(x) - mu, 0)


def compute_objective(X, z_hat, D, reg):
    res = (X - reconstruct(z_hat, D)).ravel()
    return 0.5 * np.dot(res, res) + reg * abs(z_hat).sum()


def _is_rank1(D):
    return isinstance(D, tuple)


def reconstruct(z_hat, D):
    """Convolve z_hat and D for rank-1 and full rank cases.

    z_hat : array, shape (n_atoms, *valid_support)
        Activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, *atom_support) or rank 1 with
        a tuple with shapes (n_atoms, n_channels) and
        (n_atoms, *atom_support).
    """
    if _is_rank1(D):
        u, v = D
        assert z_hat.shape[0] == u.shape[0] == v.shape[0]
        return _dense_convolve_multi_uv(z_hat, uv=D)
    else:
        assert z_hat.shape[0] == D.shape[0]
        return _dense_convolve_multi(z_hat, D)


def _dense_convolve_multi(z_hat, D):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    return np.sum([[signal.fftconvolve(zk, dkp) for dkp in dk]
                   for zk, dk in zip(z_hat, D)], 0)


def _dense_convolve_multi_uv(z_hat, uv):
    """Convolve z_hat[k] and uv[k] for each atom k, and return the sum.

    z_hat : array, shape (n_atoms, *valid_support)
        Activations
    uv : (array, array) tuple, shapes (n_atoms, n_channels) and
         (n_atoms, *atom_support)
        The atoms.
    """
    u, v = uv
    n_channels, = u.shape[1:]
    n_atoms, *valid_support = z_hat.shape
    n_atoms, *atom_support = v.shape

    Xi = np.zeros((n_channels, *get_full_support(valid_support, atom_support)))

    for zik, uk, vk in zip(z_hat, u, v):
        zik_vk = signal.fftconvolve(zik, vk)
        # Add a new dimension for each dimension in atom_support to uk
        uk = uk.reshape(*uk.shape, *(1,) * len(atom_support))
        Xi += zik_vk[None, :] * uk

    return Xi


def _dense_transpose_convolve(residual_i, D):
    """Convolve residual[i] with the transpose for each atom k

    Parameters
    ----------
    residual_i : array, shape (n_channels, *signal_support)
    D : array, shape (n_atoms, n_channels, n_times_atom) or
        tuple(array), shape (n_atoms, n_channels) x (n_atoms, *atom_support)

    Return
    ------
    grad_zi : array, shape (n_atoms, n_times_valid)

    """

    if _is_rank1(D):
        u, v = D
        flip_axis = tuple(range(1, v.ndim))
        # multiply by the spatial filter u
        # shape (n_atoms, *atom_support))
        uR_i = np.tensordot(u, residual_i, (1, 0))

        # Now do the dot product with the transpose of D (D.T) which is
        # the conv by the reversed filter (keeping valid mode)
        return np.array([
            signal.fftconvolve(uR_ik, v_k, mode='valid')
            for (uR_ik, v_k) in zip(uR_i, np.flip(v, flip_axis))
        ])
    else:
        flip_axis = tuple(range(2, D.ndim))
        return np.sum([[signal.fftconvolve(res_ip, d_kp, mode='valid')
                        for res_ip, d_kp in zip(residual_i, d_k)]
                       for d_k in np.flip(D, flip_axis)], axis=1)
