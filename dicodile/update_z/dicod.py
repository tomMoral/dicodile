"""Convolutional Sparse Coding with DICOD

Author : tommoral <thomas.moreau@inria.fr>
"""

import time
import logging
import numpy as np
from mpi4py import MPI

from ..utils import constants
from ..utils import debug_flags as flags
from ..utils.csc import _is_rank1, compute_objective
from ..utils.debugs import main_check_beta
from .coordinate_descent import STRATEGIES
from ..utils.segmentation import Segmentation
from .coordinate_descent import coordinate_descent
from ..utils.mpi import broadcast_array, recv_reduce_sum_array
from ..utils.shape_helpers import get_valid_support, find_grid_size

from ..workers.mpi_workers import MPIWorkers


log = logging.getLogger('dicod')

# debug flags

interactive_exec = "xterm"
interactive_args = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


def dicod(X_i, D, reg, z0=None, DtD=None, n_seg='auto', strategy='greedy',
          soft_lock='border', n_workers=1, w_world='auto', hostfile=None,
          tol=1e-5, max_iter=100000, timeout=None, z_positive=False,
          return_ztz=False, warm_start=False, freeze_support=False,
          timing=False, random_state=None, verbose=0, debug=False):
    """DICOD for 2D convolutional sparse coding.

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, *sig_support)
        Image to encode on the dictionary D
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        Current dictionary for the sparse coding
    reg : float
        Regularization parameter
    z0 : ndarray, shape (n_atoms, *valid_support) or None
        Warm start value for z_hat. If None, z_hat is initialized to 0.
    DtD : ndarray, shape (n_atoms, n_atoms, 2 valid_support - 1) or None
        Warm start value for DtD. If None, it is computed in each worker.
    n_seg : int or {{ 'auto' }}
        Number of segments to use for each dimension. If set to 'auto' use
        segments of twice the size of the dictionary.
    strategy : str in {}
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy', the coordinate with the largest value for dz_opt is selected.
        If set to 'random', the coordinate is chosen uniformly on the segment.
    soft_lock : str in {{ 'none', 'corner', 'border' }}
        If set to true, use the soft-lock in LGCD.
    n_workers : int
        Number of workers used to compute the convolutional sparse coding
        solution.
    w_world : int or {{'auto'}}
        Number of jobs used per row in the splitting grid. This should divide
        n_workers.
    hostfile : str
        File containing the cluster information. See MPI documentation to have
        the format of this file.
    tol : float
        Tolerance for the minimal update size in this algorithm.
    max_iter : int
        Maximal number of iteration run by this algorithm.
    timeout : int
        Timeout for the algorithm in seconds
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    return_ztz : boolean
        If True, returns the constants ztz and ztX, used to compute D-updates.
    warm_start : boolean
        If set to True, start from the previous solution z_hat if it exists.
    freeze_support : boolean
        If set to True, only update the coefficient that are non-zero in z0.
    timing : boolean
        If set to True, log the cost and timing information.
    random_state : None or int or RandomState
        current random state to seed the random number generator.
    verbose : int
        Verbosity level of the algorithm.

    Return
    ------
    z_hat : ndarray, shape (n_atoms, *valid_support)
        Activation associated to X_i for the given dictionary D
    """
    if strategy == 'lgcd':
        strategy = 'greedy'
        assert n_seg == 'auto', "strategy='lgcd' only work with n_seg='auto'."
    elif strategy == 'gcd':
        strategy = 'greedy'
        assert n_seg == 'auto', "strategy='gcd' only work with n_seg='auto'."
        n_seg = 1

    # Parameters validation
    n_channels, *sig_support = X_i.shape
    n_atoms, n_channels, *atom_support = D.shape
    assert D.ndim - 1 == X_i.ndim
    valid_support = get_valid_support(sig_support, atom_support)

    assert soft_lock in ['none', 'corner', 'border']
    assert strategy in ['greedy', 'random', 'cyclic', 'cyclic-r']

    if n_workers == 1:
        return coordinate_descent(
            X_i, D, reg, z0=z0, DtD=DtD, n_seg=n_seg, strategy=strategy,
            tol=tol, max_iter=max_iter, timeout=timeout, z_positive=z_positive,
            freeze_support=freeze_support, return_ztz=return_ztz,
            timing=timing, random_state=random_state, verbose=verbose)

    params = dict(
        strategy=strategy, tol=tol, max_iter=max_iter, timeout=timeout,
        n_seg=n_seg, z_positive=z_positive, verbose=verbose, timing=timing,
        debug=debug, random_state=random_state, reg=reg, return_ztz=return_ztz,
        soft_lock=soft_lock, precomputed_DtD=DtD is not None,
        freeze_support=freeze_support, warm_start=warm_start,
        rank1=_is_rank1(D)
    )

    workers = _spawn_workers(n_workers, hostfile)
    t_transfer, workers_segments = _send_task(workers, X_i,
                                              D, z0, DtD, w_world,
                                              params)

    if flags.CHECK_WARM_BETA:
        main_check_beta(workers.comm, workers_segments)

    if verbose > 0:
        print('\r[INFO:DICOD-{}] End transfer - {:.4}s'
              .format(workers_segments.effective_n_seg, t_transfer).ljust(80))

    # Wait for the result computation
    workers.comm.Barrier()
    run_statistics = _gather_run_statistics(
        workers.comm, workers_segments, verbose=verbose)

    z_hat, ztz, ztX, cost, _log, t_reduce = _recv_result(
        workers.comm, D.shape, valid_support, workers_segments,
        return_ztz=return_ztz, timing=timing, verbose=verbose)
    workers.comm.Barrier()

    if timing:
        p_obj = reconstruct_pobj(X_i, D, reg, _log, t_transfer, t_reduce,
                                 n_workers=n_workers,
                                 valid_support=valid_support, z0=z0)
    else:
        p_obj = [[run_statistics['n_updates'],
                  run_statistics['runtime'],
                  cost]]

    return z_hat, ztz, ztX, p_obj, run_statistics


def reconstruct_pobj(X, D, reg, _log, t_init, t_reduce, n_workers,
                     valid_support=None, z0=None):
    n_atoms = D.shape[0]
    if z0 is None:
        z_hat = np.zeros((n_atoms, *valid_support))
    else:
        z_hat = np.copy(z0)

    # Re-order the updates
    _log.sort()
    max_ii = [0] * n_workers
    for _, ii, rank, *_ in _log:
        max_ii[rank] = max(max_ii[rank], ii)
    max_ii = np.sum(max_ii)

    up_ii = 0
    p_obj = [(up_ii, t_init, compute_objective(X, z_hat, D, reg))]
    next_ii_cost = 1
    last_ii = [0] * n_workers
    for i, (t_update, ii, rank, k0, pt0, dz) in enumerate(_log):
        z_hat[k0][tuple(pt0)] += dz
        up_ii += ii - last_ii[rank]
        last_ii[rank] = ii
        if up_ii >= next_ii_cost:
            p_obj.append((up_ii, t_update + t_init,
                          compute_objective(X, z_hat, D, reg)))
            next_ii_cost = next_ii_cost * 1.3
            print("\rReconstructing cost {:7.2%}"
                  .format(np.log2(up_ii)/np.log2(max_ii)), end='', flush=True)
        elif i + 1 % 1000:
            print("\rReconstructing cost {:7.2%}"
                  .format(np.log2(up_ii)/np.log2(max_ii)), end='', flush=True)
    print('\rReconstruction cost: done'.ljust(40))

    final_cost = compute_objective(X, z_hat, D, reg)
    p_obj.append((up_ii, t_update, final_cost))
    p_obj.append((up_ii, t_init + t_update + t_reduce, final_cost))
    return np.array(p_obj)


def _spawn_workers(n_workers, hostfile):
    workers = MPIWorkers(n_workers, hostfile=hostfile)
    workers.send_command(constants.TAG_WORKER_RUN_DICOD)
    return workers


def _send_task(workers, X, D, z0, DtD, w_world, params):
    t_start = time.time()
    if _is_rank1(D):
        u, v = D
        atom_support = v.shape[1:]

    else:
        n_atoms, n_channels, *atom_support = D.shape

    _send_params(workers, params)

    _send_D(workers, D, DtD)

    workers_segments = _send_signal(workers, w_world, atom_support, X, z0)

    t_init = time.time() - t_start
    return t_init, workers_segments


def _send_params(workers, params):
    workers.comm.bcast(params, root=MPI.ROOT)


def _send_D(workers, D, DtD=None):
    if _is_rank1(D):
        u, v = D
        broadcast_array(workers.comm, u)
        broadcast_array(workers.comm, v)
    else:
        broadcast_array(workers.comm, D)
    if DtD is not None:
        broadcast_array(workers.comm, DtD)


def _send_signal(workers, w_world, atom_support, X, z0=None):
    n_workers = workers.comm.Get_remote_size()
    n_channels, *full_support = X.shape
    valid_support = get_valid_support(full_support, atom_support)
    overlap = tuple(np.array(atom_support) - 1)

    X_info = dict(has_z0=z0 is not None, valid_support=valid_support)

    if w_world == 'auto':
        X_info["workers_topology"] = find_grid_size(
            n_workers, valid_support, atom_support
        )
    else:
        assert n_workers % w_world == 0
        X_info["workers_topology"] = w_world, n_workers // w_world

    # compute a segmentation for the image,
    workers_segments = Segmentation(n_seg=X_info['workers_topology'],
                                    signal_support=valid_support,
                                    overlap=overlap)

    # Make sure that each worker has at least a segment of twice the size of
    # the dictionary. If this is not the case, the algorithm is not valid as it
    # is possible to have interference with workers that are not neighbors.
    worker_support = workers_segments.get_seg_support(0, inner=True)
    msg = ("The size of the support in each worker is smaller than twice the "
           "size of the atom support. The algorithm is does not converge in "
           "this condition. Reduce the number of cores.\n"
           f"worker: {worker_support}, atom: {atom_support}, "
           f"topology: {X_info['workers_topology']}")
    assert all(
        (np.array(worker_support) >= 2 * np.array(atom_support))
        | (np.array(X_info['workers_topology']) == 1)), msg

    # Broadcast the info about this signal to the
    workers.comm.bcast(X_info, root=MPI.ROOT)

    X = np.array(X, dtype='d')

    for i_seg in range(n_workers):
        if z0 is not None:
            worker_slice = workers_segments.get_seg_slice(i_seg)
            _send_array(workers.comm, i_seg, z0[worker_slice])
        seg_bounds = workers_segments.get_seg_bounds(i_seg)
        X_worker_slice = (Ellipsis,) + tuple([
            slice(start, end + size_atom_ax - 1)
            for (start, end), size_atom_ax in zip(seg_bounds, atom_support)
        ])
        _send_array(workers.comm, i_seg, X[X_worker_slice])

    # Synchronize the multiple send with a Barrier
    workers.comm.Barrier()
    return workers_segments


def _send_array(comm, dest, arr):
    comm.Send([arr.ravel(), MPI.DOUBLE],
              dest=dest, tag=constants.TAG_ROOT + dest)


def _gather_run_statistics(comm, workers_segments, verbose=0):
    n_workers = workers_segments.effective_n_seg

    if flags.CHECK_FINAL_BETA:
        main_check_beta(comm, workers_segments)

    stats = np.array(comm.gather(None, root=MPI.ROOT))
    iterations, n_coordinate_updates = np.sum(stats[:, :2], axis=0)
    runtime, t_local_init, t_run = np.max(stats[:, 2:5], axis=0)
    t_select = np.mean(stats[:, -2], axis=0)
    t_update = np.mean([s for s in stats[:, -1] if s is not None])
    if verbose > 1:
        print("\r[INFO:DICOD-{}] converged in {:.3f}s ({:.3f}s) with "
              "{:.0f} iterations ({:.0f} updates).".format(
                  n_workers, runtime, t_run, iterations, n_coordinate_updates))
    if verbose > 5:
        print(f"\r[DEBUG:DICOD-{n_workers}] t_select={t_select:.3e}s "
              f"t_update={t_update:.3e}s")
    run_statistics = dict(
        iterations=iterations, runtime=runtime, t_init=t_local_init,
        t_run=t_run, n_updates=n_coordinate_updates, t_select=t_select,
        t_update=t_update
    )
    return run_statistics


def _recv_result(comm, D_shape, valid_support, workers_segments,
                 return_ztz=False, timing=False, verbose=0):
    n_atoms, n_channels, *atom_support = D_shape

    t_start = time.time()

    z_hat = recv_z_hat(comm, n_atoms=n_atoms,
                       workers_segments=workers_segments)

    if return_ztz:
        ztz, ztX = recv_sufficient_statistics(comm, D_shape)
    else:
        ztz, ztX = None, None

    cost = recv_cost(comm)

    _log = []
    if timing:
        for i_seg in range(workers_segments.effective_n_seg):
            _log.extend(comm.recv(source=i_seg))

    t_reduce = time.time() - t_start
    if verbose >= 5:
        print('\r[DEBUG:DICOD-{}] End finalization - {:.4}s'
              .format(workers_segments.effective_n_seg, t_reduce))

    return z_hat, ztz, ztX, cost, _log, t_reduce


def recv_z_hat(comm, n_atoms, workers_segments):

    valid_support = workers_segments.signal_support

    inner = not flags.GET_OVERLAP_Z_HAT
    z_hat = np.empty((n_atoms, *valid_support), dtype='d')
    for i_seg in range(workers_segments.effective_n_seg):
        worker_support = workers_segments.get_seg_support(
            i_seg, inner=inner)
        z_worker = np.zeros((n_atoms,) + worker_support, 'd')
        comm.Recv([z_worker.ravel(), MPI.DOUBLE], source=i_seg,
                  tag=constants.TAG_ROOT + i_seg)
        worker_slice = workers_segments.get_seg_slice(
            i_seg, inner=inner)
        z_hat[worker_slice] = z_worker

    return z_hat


def recv_z_nnz(comm, n_atoms):
    return recv_reduce_sum_array(comm, n_atoms)


def recv_sufficient_statistics(comm, D_shape):
    n_atoms, n_channels, *atom_support = D_shape
    ztz_support = tuple(2 * np.array(atom_support) - 1)
    ztz = recv_reduce_sum_array(comm, (n_atoms, n_atoms, *ztz_support))
    ztX = recv_reduce_sum_array(comm, (n_atoms, n_channels, *atom_support))
    return ztz, ztX


def recv_cost(comm):
    cost = recv_reduce_sum_array(comm, 1)
    return cost[0]


def recv_max_error_patches(comm):
    max_error_patches = comm.gather(None, root=MPI.ROOT)
    return max_error_patches


# Update the docstring
dicod.__doc__.format(STRATEGIES)
