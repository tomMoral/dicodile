import numpy as np
from scipy import signal

from .csc import _is_rank1, reconstruct
from . import check_random_state
from .shape_helpers import get_valid_support


def get_max_error_patch(X, z, D, window=False, local_segments=None):
    """Get the maximal reconstruction error patch from the data as a new atom

    This idea is used for instance in [Yellin2017]

    Parameters
    ----------
    X: array, shape (n_channels, *sig_support)
        Signals encoded in the CSC.
    z: array, shape (n_atoms, *valid_support)
        Current estimate of the coding signals.
    D: array, shape (n_atoms, *atom_support)
        Current estimate of the dictionary.
    window: boolean
        If set to True, return the patch with the largest windowed error.

    Return
    ------
    uvk: array, shape (n_channels + n_times_atom,)
        New atom for the dictionary, chosen as the chunk of data with the
        maximal reconstruction error.

    [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
    IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
    """
    atom_support = D.shape[2:]
    patch_rec_error, X = _patch_reconstruction_error(
        X, z, D, window=window, local_segments=local_segments
    )
    i0 = patch_rec_error.argmax()
    pt0 = np.unravel_index(i0, patch_rec_error.shape)

    d0_slice = tuple([slice(None)] + [
        slice(v, v + size_ax) for v, size_ax in zip(pt0, atom_support)
    ])
    d0 = X[d0_slice]

    return d0, patch_rec_error[i0]


def prox_d(D):
    sum_axis = tuple(range(1, D.ndim))
    norm_D = np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
    D /= norm_D + (norm_D <= 1e-8)
    return D


def _patch_reconstruction_error(X, z, D, window=False, local_segments=None):
    """Return the reconstruction error for each patches of size (P, L)."""
    n_trials, n_channels, *sig_support = X.shape
    atom_support = D.shape[2:]

    X_hat = reconstruct(z, D)

    # When computing a distributed patch reconstruction error,
    # we take the bounds into account.
    # ``local_segments=None`` is used when computing the reconstruction
    # error on the full signal.
    if local_segments is not None:
        X_slice = (Ellipsis,) + tuple([
            slice(start, end + size_atom_ax - 1)
            for (start, end), size_atom_ax in zip(
                local_segments.inner_bounds, atom_support)
        ])
        X, X_hat = X[X_slice], X_hat[X_slice]

    diff = (X - X_hat)
    diff *= diff

    if window:
        patch = tukey_window(atom_support)
    else:
        patch = np.ones(atom_support)

    if D.ndim == 3:
        convolution_op = np.convolve
    else:
        convolution_op = signal.convolve

    return np.sum([convolution_op(patch, diff_p, mode='valid')
                   for diff_p in diff], axis=0), X


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


def _get_patch(X, pt, atom_support):
    patch_slice = tuple([Ellipsis] + [
        slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
    return X[patch_slice]


def init_dictionary(X, n_atoms, atom_support, random_state=None):
    rng = check_random_state(random_state)

    X_std = X.std()
    n_channels, *sig_support = X.shape
    valid_support = get_valid_support(sig_support, atom_support)
    n_patches = np.product(valid_support)

    indices = iter(rng.choice(n_patches, size=10 * n_atoms, replace=False))
    D = np.empty(shape=(n_atoms, n_channels, *atom_support))
    for k in range(n_atoms):
        pt = np.unravel_index(next(indices), valid_support)
        patch = _get_patch(X, pt, atom_support)
        while np.linalg.norm(patch.ravel()) < 1e-1 * X_std:
            pt = np.unravel_index(next(indices), valid_support)
            patch = _get_patch(X, pt, atom_support)
        D[k] = patch

    D = prox_d(D)

    return D


def compute_norm_atoms(D):
    """Compute the norm of the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        Current dictionary for the sparse coding
    """
    # Average over the channels and sum over the size of the atom
    sum_axis = tuple(range(1, D.ndim))
    norm_atoms = np.sum(D * D, axis=sum_axis, keepdims=True)
    norm_atoms += (norm_atoms == 0)
    return norm_atoms[:, 0]


def compute_norm_atoms_from_DtD(DtD, n_atoms, atom_support):
    t0 = np.array(atom_support) - 1
    return np.array([DtD[(k, k, *t0)] for k in range(n_atoms)])


def norm_atoms_from_DtD_reshaped(DtD, n_atoms, atom_support):
    norm_atoms = compute_norm_atoms_from_DtD(DtD, n_atoms, atom_support)
    return norm_atoms.reshape(*norm_atoms.shape, *[1 for _ in atom_support])


def compute_DtD(D):
    """Compute the transpose convolution between the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        or (u, v) tuple of ndarrays, shapes
        (n_atoms, n_channels) x (n_atoms, *atom_support)
        Current dictionary for the sparse coding
    """
    if _is_rank1(D):
        u, v = D
        return _compute_DtD_uv(u, v)
    else:
        return _compute_DtD_D(D)


def _compute_DtD_D(D):
    # Average over the channels
    flip_axis = tuple(range(2, D.ndim))
    DtD = np.sum([[[signal.fftconvolve(di_p, dj_p, mode='full')
                    for di_p, dj_p in zip(di, dj)]
                   for dj in D]
                  for di in np.flip(D, axis=flip_axis)], axis=2)
    return DtD


def _compute_DtD_uv(u, v):
    n_atoms = v.shape[0]
    atom_support = v.shape[1:]
    # Compute vtv using `_compute_DtD_D` as if `n_channels=1`
    vtv = _compute_DtD_D(v.reshape(n_atoms, 1, *atom_support))

    # Compute the channel-wise correlation and
    # resize it for broadcasting
    uut = u @ u.T
    uut = uut.reshape(*uut.shape, *[1 for _ in atom_support])
    return vtv * uut


def tukey_window(atom_support):
    """Return a 2D tukey window to force the atoms to have 0 border."""
    tukey_window_ = np.ones(atom_support)
    for i, ax_shape in enumerate(atom_support):
        broadcast_idx = [None] * len(atom_support)
        broadcast_idx[i] = slice(None)
        tukey_window_ *= signal.tukey(ax_shape)[tuple(broadcast_idx)]
    tukey_window_ += 1e-9 * (tukey_window_ == 0)
    return tukey_window_


def get_D(u, v):
    """Compute the rank-1 dictionary associated with u and v

    Parameters
    ----------
    u: array (n_atoms, n_channels)
    v: array (n_atoms, *atom_support)

    Return
    ------
    D: array (n_atoms, n_channels, *atom_support)
    """
    n_atoms, *atom_support = v.shape
    u = u.reshape(*u.shape, *[1 for _ in atom_support])
    v = v.reshape(n_atoms, 1, *atom_support)
    return u*v


def D_shape(D):
    """
    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        or (u, v) tuple of ndarrays, shapes
        (n_atoms, n_channels) x (n_atoms, *atom_support)
        Current dictionary for the sparse coding
    """
    if _is_rank1(D):
        return _d_shape_from_uv(*D)
    else:
        return D.shape


def _d_shape_from_uv(u, v):
    """
    Parameters
    ----------
    u: ndarray, shape (n_atoms, n_channels)
    v: ndarray, shape (n_atoms, *atom_support)

    Return
    ------
    (n_atoms, n_channels, *atom_support)
    """
    return (*u.shape, *v.shape[1:])
