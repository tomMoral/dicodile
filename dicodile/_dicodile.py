import time
import numpy as np


from .update_d.update_d import update_d
from .utils.dictionary import prox_d
from .utils.dictionary import get_lambda_max

from .update_z.distributed_sparse_encoder import DistributedSparseEncoder


DEFAULT_DICOD_KWARGS = dict(max_iter=int(1e8), n_seg='auto',
                            strategy='greedy', timeout=None,
                            soft_lock='border', debug=False)


def dicodile(X, D_init, reg=.1, n_iter=100, eps=1e-5, window=False,
             z_positive=True, n_workers=4, w_world='auto',
             tol=1e-3, hostfile=None, dicod_kwargs={},
             random_state=None, verbose=0):
    r"""Convolutional dictionary learning.

    Computes a sparse representation of a signal X, returning a dictionary
    D and a sparse activation signal Z such that X is close to
    :math:`Z \ast D`.

    This is done by solving the following optimization problem:

    .. math::
        \underset{Z,D, \left \| D_{k}\right \|\leq 1}{min} \frac{1}{2}
        \left \| X - Z \ast D \right\|_{2}^{2} +
        \lambda \left \| Z \right \|_{1}

    The support for X is noted sig_support.

    The support for D is noted atom_support.

    Parameters
    ----------
    X : ndarray, shape (n_channels, *sig_support)
        Signal to encode.

        For example, a 3-channel RGB image of definition 640x480 would
        have a shape of (3, 640, 480), a grayscale image of the same definition
        would have a shape of (1, 640, 480), a single time series would have a
        shape of (1, number_of_samples)
    D_init : ndarray, shape (n_atoms, n_channels, *atom_support)
        Current atoms dictionary.
    reg : float, defaults to .1
        Regularization parameter, in [0,1]
        The λ parameter is computed as reg * lambda_max
    n_iter : int, defaults to 100
        Maximum number of iterations
    eps : float, defaults to 1e-5
        Tolerance for the stopping criterion. A lower value will result in
        more computing time.
    window : bool
        If set to True, the learned atoms are multiplied by a Tukey
        window that sets its borders to 0. This can help having patterns
        localized in the middle of the atom support and reduces
        border effects.
    z_positive : bool, default True
        If True, adds a constraint that the activations Z must be positive.
    n_workers : int, defaults to 4
        Number of workers used to compute the convolutional sparse coding
        solution.
    w_world : int or {{'auto'}}
        Number of jobs used per row in the splitting grid. This should divide
        n_workers.
    tol : float, defaults to 1e-3
        Tolerance for minimal update size.
    hostfile : str or None
        MPI hostfile as used by `mpirun`. See your MPI implementation
        documentation. Defaults to None.
    dicod_kwargs : dict
        Extra arguments passed to the dicod function.
        See `dicodile.update_z.dicod`
    random_state : None or int or RandomState
        Random state to seed the random number generator.
    verbose : int, defaults to 0
        Verbosity level, higher is more verbose.

    Returns
    -------
    D_hat : ndarray, shape (n_channels, *sig_support)
        Updated atoms dictionary.
    Z_hat : ndarray, shape (n_channels, *valid_support)
        Activations of the different atoms
        (where or when the atoms are estimated).
    pobj : list of float
        list of costs
    times : list of float
        list of running times (seconds) for each dictionary
        and activation update step.
        The total running time of the algorithm is given by
        sum(times)

    See Also
    --------
    dicodile.update_z.dicod : Convolutional sparse coding.
    """

    assert X.ndim == len(D_init.shape[2:]) + 1, \
        "Signal and Dictionary dimensions are mismatched"

    name = "DICODILE"
    lmbd_max = get_lambda_max(X, D_init).max()
    if verbose > 5:
        print("[DEBUG:{}] Lambda_max = {}".format(name, lmbd_max))

    # Scale reg and tol
    reg_ = reg * lmbd_max
    tol = (1 - reg) * lmbd_max * tol

    params = DEFAULT_DICOD_KWARGS.copy()
    params.update(dicod_kwargs)
    params.update(dict(
        z_positive=z_positive, tol=tol,
        random_state=random_state, reg=reg_, timing=False,
        return_ztz=False, freeze_support=False, warm_start=True,
    ))

    encoder = DistributedSparseEncoder(n_workers, w_world=w_world,
                                       hostfile=hostfile, verbose=verbose-1)
    encoder.init_workers(X, D_init, reg_, params)
    D_hat = D_init.copy()
    n_atoms, n_channels, *_ = D_init.shape

    # Initialize constants for computations of the dictionary gradient.
    constants = {}
    constants['n_channels'] = n_channels
    constants['XtX'] = np.dot(X.ravel(), X.ravel())

    # monitor cost function
    times = [encoder.t_init]
    pobj = [encoder.get_cost()]
    t_start = time.time()

    # Initial step_size
    step_size = 1

    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            msg = '.' if ((ii + 1) % 10 != 0) else '+\n'
            print(msg, end='', flush=True)
        elif verbose > 1:
            print('[INFO:{}] - CD iterations {} / {} ({:.0f}s)'
                  .format(name, ii, n_iter, time.time() - t_start))

        if verbose > 5:
            print('[DEBUG:{}] lambda = {:.3e}'.format(name, reg_))

        # Compute z update
        t_start_update_z = time.time()
        encoder.process_z_hat()
        times.append(time.time() - t_start_update_z)

        # monitor cost function
        pobj.append(encoder.get_cost())
        if verbose > 5:
            print('[DEBUG:{}] Objective (z) : {:.3e} ({:.0f}s)'
                  .format(name, pobj[-1], times[-1]))

        z_nnz = encoder.get_z_nnz()
        if np.all(z_nnz == 0):
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atoms has"
                          " been learned.", UserWarning)
            break

        # Compute D update
        t_start_update_d = time.time()
        constants['ztz'], constants['ztX'] = \
            encoder.get_sufficient_statistics()
        step_size *= 100
        D_hat, step_size = update_d(X, None, D_hat,
                                    constants=constants, window=window,
                                    step_size=step_size, max_iter=100,
                                    eps=1e-5, verbose=verbose, momentum=False)
        times.append(time.time() - t_start_update_d)

        # If an atom is un-used, replace it by the chunk of the residual with
        # the largest un-captured variance.
        null_atom_indices = np.where(z_nnz == 0)[0]
        if len(null_atom_indices) > 0:
            k0 = null_atom_indices[0]
            d0 = encoder.compute_and_get_max_error_patch(window=window)
            D_hat[k0] = prox_d(d0)
            if verbose > 1:
                print('[INFO:{}] Resampled atom {}'.format(name, k0))

        # Update the dictionary D_hat in the encoder
        encoder.set_worker_D(D_hat)

        # monitor cost function
        pobj.append(encoder.get_cost())
        if verbose > 5:
            print('[DEBUG:{}] Objective (d) : {:.3e}  ({:.0f}s)'
                  .format(name, pobj[-1], times[-1]))

        # Only check that the cost is always going down when the regularization
        # parameter is fixed.
        dz = (pobj[-3] - pobj[-2]) / min(pobj[-3], pobj[-2])
        du = (pobj[-2] - pobj[-1]) / min(pobj[-2], pobj[-1])
        if (dz < eps or du < eps):
            if dz < 0:
                raise RuntimeError(
                    "The z update have increased the objective value by {}."
                    .format(dz)
                )
            if du < -1e-10 and dz > 1e-12:
                raise RuntimeError(
                    "The d update have increased the objective value by {}."
                    "(dz={})".format(du, dz)
                )
            if dz < eps and du < eps:
                if verbose == 1:
                    print("")
                print("[INFO:{}] Converged after {} iteration, (dz, du) "
                      "= {:.3e}, {:.3e}".format(name, ii + 1, dz, du))
                break

    encoder.process_z_hat()
    z_hat = encoder.get_z_hat()
    pobj.append(encoder.get_cost())

    runtime = np.sum(times)

    encoder.release_workers()
    encoder.shutdown_workers()

    print("[INFO:{}] Finished in {:.0f}s".format(name, runtime))
    return D_hat, z_hat, pobj, times
