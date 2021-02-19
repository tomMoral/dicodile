import time
import numpy as np


from .update_d.update_d import update_d
from .utils.dictionary import get_lambda_max
from .utils.dictionary import get_max_error_dict

from .update_z.distributed_sparse_encoder import DistributedSparseEncoder


DEFAULT_DICOD_KWARGS = dict(max_iter=int(1e8), timeout=None)


def dicodile(X, D_hat, reg=.1, z_positive=True, n_iter=100, strategy='greedy',
             n_seg='auto', tol=1e-3, dicod_kwargs={}, stopping_pobj=None,
             w_world='auto', n_workers=4, hostfile=None, eps=1e-5,
             window=False, raise_on_increase=True, random_state=None,
             name="DICODILE", verbose=0):

    lmbd_max = get_lambda_max(X, D_hat).max()
    if verbose > 5:
        print("[DEBUG:DICODILE] Lambda_max = {}".format(lmbd_max))

    # Scale reg and tol
    reg_ = reg * lmbd_max
    tol = (1 - reg) * lmbd_max * tol

    params = DEFAULT_DICOD_KWARGS.copy()
    params.update(dicod_kwargs)
    params.update(dict(
        strategy=strategy, n_seg=n_seg, z_positive=z_positive, tol=tol,
        random_state=random_state, reg=reg_, timing=False, soft_lock='border',
        return_ztz=False, freeze_support=False, warm_start=True, debug=False
    ))

    encoder = DistributedSparseEncoder(n_workers, w_world=w_world,
                                       hostfile=hostfile, verbose=verbose-1)
    encoder.init_workers(X, D_hat, reg_, params)

    n_atoms, n_channels, *_ = D_hat.shape

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
            z_hat = encoder.get_z_hat()
            D_hat[k0] = get_max_error_dict(X, z_hat, D_hat, window=window)[0]
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
            if dz < 0 and raise_on_increase:
                raise RuntimeError(
                    "The z update have increased the objective value by {}."
                    .format(dz)
                )
            if du < -1e-10 and dz > 1e-12 and raise_on_increase:
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

        if stopping_pobj is not None and pobj[-1] < stopping_pobj:
            break

    encoder.process_z_hat()
    z_hat = encoder.get_z_hat()
    pobj.append(encoder.get_cost())

    runtime = np.sum(times)
    encoder.release_workers()
    print("[INFO:{}] Finished in {:.0f}s".format(name, runtime))
    return pobj, times, D_hat, z_hat
