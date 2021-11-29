"""Convolutional Sparse Coding with LGCD

Author : tommoral <thomas.moreau@inria.fr>
"""

import time
import numpy as np


from dicodile.utils.csc import _dense_transpose_convolve, reconstruct
from dicodile.utils import check_random_state
from dicodile.utils import debug_flags as flags
from dicodile.utils.segmentation import Segmentation
from dicodile.utils.csc import compute_ztz, compute_ztX
from dicodile.utils.shape_helpers import get_valid_support
from dicodile.utils.order_iterator import get_order_iterator
from dicodile.utils.csc import compute_objective, soft_thresholding
from dicodile.utils.dictionary import D_shape, compute_DtD,\
    compute_norm_atoms, norm_atoms_from_DtD_reshaped


STRATEGIES = {'greedy', 'random', 'cyclic', 'cyclic-r', 'gs-r', 'gs-q'}


def coordinate_descent(X_i, D, reg, z0=None, DtD=None, n_seg='auto',
                       strategy='greedy', tol=1e-5, max_iter=100000,
                       timeout=None, z_positive=False, freeze_support=False,
                       return_ztz=False, timing=False,
                       random_state=None, verbose=0):
    """Coordinate Descent Algorithm for 2D convolutional sparse coding.

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, *sig_support)
        Image to encode on the dictionary D
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        Current dictionary for the sparse coding
    reg : float
        Regularization parameter
    z0 : ndarray, shape (n_atoms, *valid_support) or None
        Warm start value for z_hat. If not present, z_hat is initialized to 0.
    DtD : ndarray, shape (n_atoms, n_atoms, 2 * valid_support - 1) or None
        Warm start value for DtD. If not present, it is computed on init.
    n_seg : int or 'auto'
        Number of segments to use for each dimension. If set to 'auto' use
        segments of twice the size of the dictionary.
    strategy : str in {strategies}
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy'|'gs-r', the coordinate with the largest value for dz_opt is
        selected. If set to 'random, the coordinate is chosen uniformly on the
        segment. If set to 'gs-q', the value that reduce the most the cost
        function is selected. In this case, dE must holds the value of this
        cost reduction.
    tol : float
        Tolerance for the minimal update size in this algorithm.
    max_iter : int
        Maximal number of iteration run by this algorithm.
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    freeze_support : boolean
        If set to True, only update the coefficient that are non-zero in z0.
    return_ztz : boolean
        If True, returns the constants ztz and ztX, used to compute D-updates.
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
    n_channels, *sig_support = X_i.shape
    n_atoms, n_channels, *atom_support = D.shape
    valid_support = get_valid_support(sig_support, atom_support)

    if strategy not in STRATEGIES:
        raise ValueError("'The coordinate selection strategy should be in "
                         "{}. Got '{}'.".format(STRATEGIES, strategy))

    # compute sizes for the segments for LGCD. Auto gives segments of size
    # twice the support of the atoms.
    if n_seg == 'auto':
        n_seg = np.array(valid_support) // (2 * np.array(atom_support) - 1)
        n_seg = tuple(np.maximum(1, n_seg))
    segments = Segmentation(n_seg, signal_support=valid_support)

    # Pre-compute constants for maintaining the auxillary variable beta and
    # compute the coordinate update values.
    constants = {}
    constants['norm_atoms'] = compute_norm_atoms(D)
    if DtD is None:
        constants['DtD'] = compute_DtD(D)
    else:
        constants['DtD'] = DtD

    # Initialization of the algorithm variables
    i_seg = -1
    accumulator = 0
    if z0 is None:
        z_hat = np.zeros((n_atoms,) + valid_support)
    else:
        z_hat = np.copy(z0)
    n_coordinates = z_hat.size

    # Get a random number genator from the given random_state
    rng = check_random_state(random_state)
    order = None
    if strategy in ['cyclic', 'cyclic-r', 'random']:
        order = get_order_iterator(z_hat.shape, strategy=strategy,
                                   random_state=rng)

    t_start_init = time.time()
    return_dE = strategy == "gs-q"
    beta, dz_opt, dE = _init_beta(X_i, D, reg, z_i=z0, constants=constants,
                                  z_positive=z_positive, return_dE=return_dE)
    if strategy == "gs-q":
        raise NotImplementedError("This is still WIP")

    if freeze_support:
        freezed_support = z0 == 0
        dz_opt[freezed_support] = 0
    else:
        freezed_support = None

    p_obj, next_log_iter = [], 1
    t_init = time.time() - t_start_init
    if timing:
        p_obj.append((0, t_init, 0, compute_objective(X_i, z_hat, D, reg)))

    n_coordinate_updates = 0
    t_run = 0
    t_select_coord, t_update_coord = [], []
    t_start = time.time()
    if timeout is not None:
        deadline = t_start + timeout
    else:
        deadline = None
    for ii in range(max_iter):
        if ii % 1000 == 0 and verbose > 0:
            print("\r[LGCD:PROGRESS] {:.0f}s - {:7.2%} iterations"
                  .format(t_run, ii / max_iter), end='', flush=True)

        i_seg = segments.increment_seg(i_seg)
        if segments.is_active_segment(i_seg):
            t_start_selection = time.time()
            k0, pt0, dz = _select_coordinate(dz_opt, dE, segments, i_seg,
                                             strategy=strategy, order=order)
            selection_duration = time.time() - t_start_selection
            t_select_coord.append(selection_duration)
            t_run += selection_duration
        else:
            dz = 0

        accumulator = max(abs(dz), accumulator)

        # Update the selected coordinate and beta, only if the update is
        # greater than the convergence tolerance.
        if abs(dz) > tol:
            t_start_update = time.time()

            # update the current solution estimate and beta
            beta, dz_opt, dE = coordinate_update(
                k0, pt0, dz, beta=beta, dz_opt=dz_opt, dE=dE, z_hat=z_hat, D=D,
                reg=reg, constants=constants, z_positive=z_positive,
                freezed_support=freezed_support)
            touched_segs = segments.get_touched_segments(
                pt=pt0, radius=atom_support)
            n_changed_status = segments.set_active_segments(touched_segs)

            # Logging of the time and the cost function if necessary
            update_duration = time.time() - t_start_update
            n_coordinate_updates += 1
            t_run += update_duration
            t_update_coord.append(update_duration)
            if timing and ii + 1 >= next_log_iter:
                p_obj.append((ii + 1, t_run, np.sum(t_select_coord),
                              compute_objective(X_i, z_hat, D, reg)))
                next_log_iter = next_log_iter * 1.3

            # If debug flag CHECK_ACTIVE_SEGMENTS is set, check that all
            # inactive segments should be inactive
            if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
                segments.test_active_segment(dz_opt, tol)

        elif strategy in ["greedy", 'gs-r']:
            segments.set_inactive_segments(i_seg)

        # check stopping criterion
        if _check_convergence(segments, tol, ii, dz_opt, n_coordinates,
                              strategy, accumulator=accumulator):
            assert np.all(abs(dz_opt) <= tol)
            if verbose > 0:
                print("\r[LGCD:INFO] converged in {} iterations ({} updates)"
                      .format(ii + 1, n_coordinate_updates))

            break

        # Check is we reach the timeout
        if deadline is not None and time.time() >= deadline:
            if verbose > 0:
                print("\r[LGCD:INFO] Reached timeout. Done {} iterations "
                      "({} updates). Max of |dz|={}."
                      .format(ii + 1, n_coordinate_updates, abs(dz_opt).max()))
            break
    else:
        if verbose > 0:
            print("\r[LGCD:INFO] Reached max_iter. Done {} coordinate "
                  "updates. Max of |dz|={}."
                  .format(n_coordinate_updates, abs(dz_opt).max()))

    print(f"\r[LGCD:{strategy}] "
          f"t_select={np.mean(t_select_coord):.3e}s  "
          f"t_update={np.mean(t_update_coord):.3e}s"
          )

    runtime = time.time() - t_start
    if verbose > 0:
        print("\r[LGCD:INFO] done in {:.3f}s ({:.3f}s)"
              .format(runtime, t_run))

    ztz, ztX = None, None
    if return_ztz:
        ztz = compute_ztz(z_hat, atom_support)
        ztX = compute_ztX(z_hat, X_i)

    p_obj.append([n_coordinate_updates, t_run,
                  compute_objective(X_i, z_hat, D, reg)])

    run_statistics = dict(iterations=ii + 1, runtime=runtime, t_init=t_init,
                          t_run=t_run, n_updates=n_coordinate_updates,
                          t_select=np.mean(t_select_coord),
                          t_update=np.mean(t_update_coord))

    return z_hat, ztz, ztX, p_obj, run_statistics


def _init_beta(X_i, D, reg, z_i=None, constants={}, z_positive=False,
               return_dE=False):
    """Init beta with the gradient in the current point 0

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, *sig_support)
        Image to encode on the dictionary D
    z_i : ndarray, shape (n_atoms, *valid_support)
        Warm start value for z_hat
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        Current dictionary for the sparse coding
    reg : float
        Regularization parameter
    constants : dictionary, optional
        Pre-computed constants for the computations
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    return_dE : boolean
        If set to true, return a vector holding the value of cost update when
        updating coordinate i to value dz_opt[i].
    """
    if 'norm_atoms' in constants:
        norm_atoms = constants['norm_atoms']
    else:
        norm_atoms = compute_norm_atoms(D)

    if z_i is not None and abs(z_i).sum() > 0:
        residual = reconstruct(z_i, D) - X_i
    else:
        residual = -X_i

    beta = _dense_transpose_convolve(residual_i=residual, D=D)

    if z_i is not None:
        assert z_i.shape == beta.shape
        for k, *pt in zip(*z_i.nonzero()):
            pt = tuple(pt)
            beta[(k, *pt)] -= z_i[(k, *pt)] * norm_atoms[k]

    dz_opt = soft_thresholding(-beta, reg, positive=z_positive) / norm_atoms

    if z_i is not None:
        dz_opt -= z_i

    if return_dE:
        dE = compute_dE(dz_opt, beta, z_i, reg)
    else:
        dE = None

    return beta, dz_opt, dE


def _select_coordinate(dz_opt, dE, segments, i_seg, strategy, order=None):
    """Pick a coordinate to update

    Parameters
    ----------
    dz_opt : ndarray, shape (n_atoms, *valid_support)
        Difference between the current value and the optimal value for each
        coordinate.
    dE : ndarray, shape (n_atoms, *valid_support) or None
        Value of the reduction of the cost when moving a given coordinate to
        the optimal value dz_opt. This is only necessary when strategy is
        'gs-q'.
    segments : dicod.utils.Segmentation
        Segmentation info for LGCD
    i_seg : int
        Current segment indices in the Segmentation object.
    strategy : str in {strategies}
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy'|'gs-r', the coordinate with the largest value for dz_opt is
        selected. If set to 'random, the coordinate is chosen uniformly on the
        segment. If set to 'gs-q', the value that reduce the most the cost
        function is selected. In this case, dE must holds the value of this
        cost reduction.
    order : ndarray or None
        an array to store the order to select the coordinate for strategies
        cyclic-r and random.
    """

    if strategy in ['random', 'cyclic-r', 'cyclic']:
        k0, *pt0 = next(order)
    else:
        if strategy in ['greedy', 'gs-r']:
            seg_slice = segments.get_seg_slice(i_seg, inner=True)
            dz_opt_seg = dz_opt[seg_slice]
            i0 = abs(dz_opt_seg).argmax()

        elif strategy == 'gs-q':
            seg_slice = segments.get_seg_slice(i_seg, inner=True)
            dE_seg = dE[seg_slice]
            i0 = abs(dE_seg).argmax()
        # TODO: broken~~~!!!
        k0, *pt0 = np.unravel_index(i0, dz_opt_seg.shape)
        # k0, *pt0 = tuple(fast_unravel(i0, dz_opt_seg.shape))
        pt0 = segments.get_global_coordinate(i_seg, pt0)

    dz = dz_opt[(k0, *pt0)]
    return k0, pt0, dz


def coordinate_update(k0, pt0, dz, beta, dz_opt, dE, z_hat, D, reg, constants,
                      z_positive, freezed_support=None, coordinate_exist=True):
    """Update the optimal value for the coordinate updates.

    Parameters
    ----------
    k0, pt0 : int, (int, int)
        Indices of the coordinate updated.
    dz : float
        Value of the update.
    beta, dz_opt : ndarray, shape (n_atoms, *valid_support)
        Auxillary variables holding the optimal value for the coordinate update
    dE : ndarray, shape (n_atoms, *valid_support) or None
        If not None, dE[i] contains the change in cost value when the
        coordinate i is updated to value dz_opt[i].
    z_hat : ndarray, shape (n_atoms, *valid_support)
        Value of the coordinate.
    D : ndarray, shape (n_atoms, n_channels, *atom_support)
        Current dictionary for the sparse coding
    reg : float
        Regularization parameter
    constants : dictionary, optional
        Pre-computed constants for the computations
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    freezed_support : ndarray, shape (n_atoms, *valid_support)
        mask with True in each coordinate fixed to 0.
    coordinate_exist : boolean
        If set to true, the coordinate is located in the updated part of beta.
        This option is only useful for DICOD.


    Return
    ------
    beta, dz_opt : ndarray, shape (n_atoms, *valid_support)
        Auxillary variables holding the optimal value for the coordinate update
    """
    n_atoms, *valid_support = beta.shape
    n_atoms, n_channels, *atom_support = D_shape(D)

    if 'DtD' in constants:
        DtD = constants['DtD']
    else:
        DtD = compute_DtD(D)
    if 'norm_atoms' in constants:
        norm_atoms = constants['norm_atoms']
    else:
        norm_atoms = norm_atoms_from_DtD_reshaped(DtD, n_atoms, atom_support)

    # define the bounds for the beta update
    update_slice, DtD_slice = (Ellipsis,), (Ellipsis, k0)
    for v, size_atom_ax, size_valid_ax in zip(pt0, atom_support,
                                              valid_support):
        start_up_ax = max(0, v - size_atom_ax + 1)
        end_up_ax = min(size_valid_ax, v + size_atom_ax)
        update_slice = update_slice + (slice(start_up_ax, end_up_ax),)
        start_DtD_ax = max(0, size_atom_ax - 1 - v)
        end_DtD_ax = start_DtD_ax + (end_up_ax - start_up_ax)
        DtD_slice = DtD_slice + (slice(start_DtD_ax, end_DtD_ax),)

    # update the coordinate and beta
    if coordinate_exist:
        z_hat[k0][pt0] += dz
        beta_i0 = beta[k0][pt0]
    beta[update_slice] += DtD[DtD_slice] * dz

    # update dz_opt
    tmp = soft_thresholding(-beta[update_slice], reg,
                            positive=z_positive) / norm_atoms
    dz_opt[update_slice] = tmp - z_hat[update_slice]

    if freezed_support is not None:
        dz_opt[update_slice][freezed_support[update_slice]] = 0

    # If the coordinate exists, put it back to 0 update
    if coordinate_exist:
        beta[k0][pt0] = beta_i0
        dz_opt[k0][pt0] = 0

    # Update dE is needed
    if dE is not None:
        dE[update_slice] = compute_dE(dz_opt[update_slice], beta[update_slice],
                                      z_hat[update_slice], reg)

    return beta, dz_opt, dE


def compute_dE(dz_opt, beta, z_hat, reg):
    if z_hat is None:
        z_hat = 0
    return (
        # l2 term
        dz_opt * (z_hat + .5 * dz_opt - beta)
        # l1 term
        + reg * (abs(z_hat) - abs(z_hat + dz_opt))
        )


def _check_convergence(segments, tol, iteration, dz_opt, n_coordinates,
                       strategy, accumulator=0):
    """Check convergence for the coordinate descent algorithm

    Parameters
    ----------
    segments : Segmentation
        Number of active segment at this iteration.
    tol : float
        Tolerance for the minimal update size in this algorithm.
    iteration : int
        Current iteration number
    dz_opt : ndarray, shape (n_atoms, *valid_support)
        Difference between the current value and the optimal value for each
        coordinate.
    n_coordinates : int
        Number of coordinate in the considered problem.
    strategy : str in {strategies}
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy', the coordinate with the largest value for dz_opt is selected.
        If set to 'random, the coordinate is chosen uniformly on the segment.
    accumulator : float, (default: 0)
        In the case of strategy 'random', accumulator should keep track of an
        approximation of max(abs(dz_opt)). The full convergence criterion will
        only be checked if accumulator <= tol.
    """
    is_epoch = (iteration + 1) % n_coordinates == 0
    if strategy not in ['greedy', 'gs-r', 'gs-q'] and is_epoch:
        for i_seg in range(segments.effective_n_seg):
            seg_slice = segments.get_seg_slice(i_seg, inner=True)
            if np.all(abs(dz_opt[seg_slice]) <= tol):
                segments.set_inactive_segments(i_seg)

    # check stopping criterion
    return not segments.exist_active_segment()


# Set the strategies in docstring
for f in [_check_convergence, _select_coordinate, coordinate_descent]:
    f.__doc__ = f.__doc__.format(strategies=STRATEGIES)
