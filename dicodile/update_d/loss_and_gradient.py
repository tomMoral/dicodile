# Authors: Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from scipy import signal

from ..utils.shape_helpers import get_valid_support


def compute_objective(D, constants):
    """Compute the value of the objective function

    Parameters
    ----------
    D : array, shape (n_atoms, n_channels, *atom_support)
        Current dictionary
    constants : dict
        Constant to accelerate the computation when updating D.
    """
    return _l2_objective(D=D, constants=constants)


def gradient_d(D=None, X=None, z=None, constants=None,
               return_func=False, flatten=False):
    """Compute the gradient of the reconstruction loss relative to d.

    Parameters
    ----------
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    X : array, shape (n_trials, n_channels, n_times) or None
        The data array
    z : array, shape (n_atoms, n_trials, n_times_valid) or None
        The activations
    constants : dict or None
        Constant to accelerate the computation of the gradient
    return_func : boolean
        Returns also the objective function, used to speed up LBFGS solver
    flatten : boolean
        If flatten is True, takes a flatten uv input and return the gradient
        as a flatten array.

    Returns
    -------
    (func) : float
        The objective function
    grad : array, shape (n_atoms * n_times_valid)
        The gradient
    """
    if flatten:
        if z is None:
            n_channels = constants['n_channels']
            n_atoms, _, *ztz_support = constants['ztz'].shape
            atom_support = tuple((np.array(ztz_support) + 1) // 2)
        else:
            n_trial, n_channels, *sig_support = X.shape
            n_trials, n_atoms, *valid_support = z.shape
            atom_support = get_valid_support(sig_support, valid_support)
        D = D.reshape((n_atoms, n_channels, *atom_support))

    cost, grad_d = _l2_gradient_d(D=D, constants=constants,
                                  return_func=return_func)

    if flatten:
        grad_d = grad_d.ravel()

    if return_func:
        return cost, grad_d

    return grad_d


def _l2_gradient_d(D, constants=None, return_func=False):

    cost = None
    assert D is not None
    g = tensordot_convolve(constants['ztz'], D)
    if return_func:
        cost = .5 * g - constants['ztX']
        cost = np.dot(D.ravel(), g.ravel())
        if 'XtX' in constants:
            cost += constants['XtX']
    return cost, g - constants['ztX']


def _l2_objective(D=None, constants=None):

    # Fast compute the l2 objective when updating uv/D
    assert D is not None, "D is needed to fast compute the objective."
    grad_d = .5 * tensordot_convolve(constants['ztz'], D)
    grad_d -= constants['ztX']
    cost = (D * grad_d).sum()

    cost += .5 * constants['XtX']
    return cost


def tensordot_convolve(ztz, D):
    """Compute the multivariate (valid) convolution of ztz and D

    Parameters
    ----------
    ztz: array, shape = (n_atoms, n_atoms, *(2 * atom_support - 1))
        Activations
    D: array, shape = (n_atoms, n_channels, atom_support)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, *atom_support)
        Gradient
    """
    n_atoms, n_channels, *atom_support = D.shape

    n_time_support = np.prod(atom_support)

    if n_time_support < 512:

        G = np.zeros(D.shape)
        axis_sum = list(range(2, D.ndim))
        D_revert = np.flip(D, axis=axis_sum)
        for t in range(n_time_support):
            pt = np.unravel_index(t, atom_support)
            ztz_slice = tuple([Ellipsis] + [
                slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
            G[(Ellipsis, *pt)] = np.tensordot(ztz[ztz_slice], D_revert,
                                              axes=([1] + axis_sum,
                                                    [0] + axis_sum))
    else:
        if D.ndim == 3:
            convolution_op = np.convolve
        else:
            convolution_op = signal.fftconvolve
        G = np.sum([[[convolution_op(ztz_kk, D_kp, mode='valid')
                    for D_kp in D_k] for ztz_kk, D_k in zip(ztz_k, D)]
                   for ztz_k in ztz], axis=1)
    return G
