import numpy as np
from scipy import signal

from .csc import reconstruct
from . import check_random_state
from .shape_helpers import get_valid_shape


def get_max_error_dict(X, z, D):
    """Get the maximal reconstruction error patch from the data as a new atom

    This idea is used for instance in [Yellin2017]

    Parameters
    ----------
    X: array, shape (n_channels, *sig_shape)
        Signals encoded in the CSC.
    z: array, shape (n_atoms, *valid_shape)
        Current estimate of the coding signals.
    D: array, shape (n_atoms, *atom_support)
        Current estimate of the dictionary.

    Return
    ------
    uvk: array, shape (n_channels + n_times_atom,)
        New atom for the dictionary, chosen as the chunk of data with the
        maximal reconstruction error.

    [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
    IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
    """
    atom_support = D.shape[2:]
    patch_rec_error = _patch_reconstruction_error(X, z, D)
    i0 = patch_rec_error.argmax()
    pt0 = np.unravel_index(i0, patch_rec_error.shape)

    d0_slice = tuple([slice(None)] + [
        slice(v, v + size_ax) for v, size_ax in zip(pt0, atom_support)
    ])
    d0 = X[d0_slice]

    d0 = prox_d(d0)

    return d0


def prox_d(D):
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
    return D


def _patch_reconstruction_error(X, z, D):
    """Return the reconstruction error for each patches of size (P, L)."""
    n_trials, n_channels, *sig_shape = X.shape
    atom_support = D.shape[2:]

    X_hat = reconstruct(z, D)

    diff = (X - X_hat)
    diff *= diff
    patch = np.ones(atom_support)

    if D.ndim == 3:
        convolution_op = np.convolve
    else:
        convolution_op = signal.convolve

    return np.sum([convolution_op(patch, diff_p, mode='valid')
                   for diff_p in diff], axis=0)


def get_lambda_max(X, D_hat):
    # multivariate general case

    if D_hat.ndim == 3:
        correlation_op = np.correlate
    else:
        correlation_op = signal.correlate

    return np.max([
        np.sum([    # sum over the channels
            correlation_op(D_kp, X_ip, mode='valid')
            for D_kp, X_ip in zip(D_k, X)
        ], axis=0) for D_k in D_hat])


def init_dictionary(X, n_atoms, atom_support, random_state=None):
    rng = check_random_state(random_state)

    n_channels, *sig_shape = X.shape
    valid_shape = get_valid_shape(sig_shape, atom_support)

    indices = np.c_[[rng.randint(size_ax, size=(n_atoms))
                     for size_ax in valid_shape]].T
    D = np.empty(shape=(n_atoms, n_channels, *atom_support))
    for k, pt in enumerate(indices):
        D_slice = tuple([Ellipsis] + [
            slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
        D[k] = X[D_slice]
    D = prox_d(D)

    return D


def compute_norm_atoms(D):
    """Compute the norm of the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    """
    # Average over the channels and sum over the size of the atom
    sum_axis = tuple(range(1, D.ndim))
    norm_atoms = np.sum(D * D, axis=sum_axis, keepdims=True)
    norm_atoms += (norm_atoms == 0)
    return norm_atoms[:, 0]


def compute_DtD(D):
    """Compute the transpose convolution between the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    """
    # Average over the channels
    flip_axis = tuple(range(2, D.ndim))
    DtD = np.sum([[[signal.fftconvolve(di_p, dj_p, mode='full')
                    for di_p, dj_p in zip(di, dj)]
                   for dj in D]
                  for di in np.flip(D, axis=flip_axis)], axis=2)
    return DtD
