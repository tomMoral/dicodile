"""Helper functions for Convolutional Sparse Coding.

Author : tommoral <thomas.moreau@inria.fr>
"""

import numpy as np
from scipy import signal

from .shape_helpers import get_valid_support


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


def reconstruct(z_hat, D):
    X_hat = np.sum([[signal.fftconvolve(z_k, d_kp) for d_kp in d_k]
                    for z_k, d_k in zip(z_hat, D)], axis=0)
    return X_hat


def compute_objective(X, z_hat, D, reg):
    res = (X - reconstruct(z_hat, D)).ravel()
    return 0.5 * np.dot(res, res) + reg * abs(z_hat).sum()
