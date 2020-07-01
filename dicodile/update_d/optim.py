import time

import numpy as np
from scipy import optimize


MIN_STEP_SIZE = 1e-10


def fista(f_obj, f_grad, f_prox, step_size, x0, max_iter, verbose=0,
          momentum=False, eps=None, adaptive_step_size=False, debug=False,
          scipy_line_search=True, name='ISTA', timing=False):
    """Proximal Gradient Descent (PGD) and Accelerated PDG.

    This reduces to ISTA and FISTA when the loss function is the l2 loss and
    the proximal operator is the soft-thresholding.

    Parameters
    ----------
    f_obj : callable
        Objective function. Used only if debug or adaptive_step_size.
    f_grad : callable
        Gradient of the objective function
    f_prox : callable
        Proximal operator
    step_size : float or None
        Step size of each update. Can be None if adaptive_step_size.
    x0 : array
        Initial point of the optimization
    max_iter : int
        Maximum number of iterations
    verbose : int
        Verbosity level
    momentum : boolean
        If True, use FISTA instead of ISTA
    eps : float or None
        Tolerance for the stopping criterion
    adaptive_step_size : boolean
        If True, the step size is adapted at each step
    debug : boolean
        If True, compute the objective function at each step and return the
        list at the end.
    timing : boolean
        If True, compute the objective function at each step, and the duration
        of each step, and return both lists at the end.

    Returns
    -------
    x_hat : array
        The final point after optimization
    pobj : list or None
        If debug is True, pobj contains the value of the cost function at each
        iteration.
    """
    obj_uv = f_obj(x0)
    pobj = None
    if debug or timing:
        pobj = [obj_uv]
    if timing:
        times = [0]
        start = time.time()

    if step_size is None:
        step_size = 1.
    if eps is None:
        eps = np.finfo(np.float32).eps

    tk = 1.0
    x_hat = x0.copy()
    x_hat_aux = x_hat.copy()
    grad = np.empty(x_hat.shape)
    diff = np.empty(x_hat.shape)
    last_up = t_start = time.time()
    has_restarted = False
    for ii in range(max_iter):
        t_update = time.time()
        if verbose > 1 and t_update - last_up > 1:
            print("\r[PROGRESS:{}] {:.0f}s - {:7.2%} iterations ({:.3e})"
                  .format(name, t_update - t_start, ii / max_iter, step_size),
                  end="", flush=True)

        grad[:] = f_grad(x_hat_aux)

        if adaptive_step_size:

            def compute_obj_and_step(step_size, return_x_hat=False):
                x_hat = f_prox(x_hat_aux - step_size * grad,
                               step_size=step_size)
                pobj = f_obj(x_hat)
                if return_x_hat:
                    return pobj, x_hat
                else:
                    return pobj

            if scipy_line_search:
                norm_grad = np.dot(grad.ravel(), grad.ravel())
                step_size, obj_uv = optimize.linesearch.scalar_search_armijo(
                    compute_obj_and_step, obj_uv, -norm_grad, c1=1e-5,
                    alpha0=step_size, amin=MIN_STEP_SIZE)
                if step_size is not None:
                    # compute the next point
                    x_hat_aux -= step_size * grad
                    x_hat_aux = f_prox(x_hat_aux, step_size=step_size)

            else:
                from functools import partial
                f = partial(compute_obj_and_step, return_x_hat=True)
                obj_uv, x_hat_aux, step_size = _adaptive_step_size(
                    f, obj_uv, alpha=step_size)

            if step_size is None or step_size < MIN_STEP_SIZE:
                # We did not find a valid step size. We should restart
                # the momentum for APGD or stop the algorithm for PDG.
                x_hat_aux = x_hat
                has_restarted = momentum and not has_restarted
                step_size = 1
                obj_uv = f_obj(x_hat)
            else:
                has_restarted = False

        else:
            x_hat_aux -= step_size * grad
            x_hat_aux = f_prox(x_hat_aux, step_size=step_size)

        diff[:] = x_hat_aux - x_hat
        x_hat[:] = x_hat_aux
        if momentum:
            tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            x_hat_aux += (tk - 1) / tk_new * diff
            tk = tk_new

        if debug or timing:
            pobj.append(f_obj(x_hat))
            if adaptive_step_size:
                assert len(pobj) < 2 or pobj[-1] <= pobj[-2]
        if timing:
            times.append(time.time() - start)
            start = time.time()

        f = np.sum(abs(diff))
        if f <= eps and not has_restarted:
            break
        if f > 1e50:
            raise RuntimeError("The D update have diverged.")
    else:
        if verbose > 1:
            print('\r[INFO:{}] update did not converge'
                  .format(name).ljust(60))
    if verbose > 1:
        print('\r[INFO:{}]: {} iterations'.format(name, ii + 1))

    if timing:
        return x_hat, pobj, times
    return x_hat, pobj, step_size


def _adaptive_step_size(f, f0=None, alpha=None, tau=2):
    """
    Parameters
    ----------
    f : callable
        Optimized function, take only the step size as argument
    f0 : float
        value of f at current point, i.e. step size = 0
    alpha : float
        Initial step size
    tau : float
        Multiplication factor of the step size during the adaptation
    """

    if alpha is None:
        alpha = 1

    if f0 is None:
        f0, _ = f(0)
    f_alpha, x_alpha = f(alpha)
    if f_alpha < f0:
        f_alpha_up, x_alpha_up = f(alpha * tau)
        if f_alpha_up < f0:
            return f_alpha_up, x_alpha_up, alpha * tau
        else:
            return f_alpha, x_alpha, alpha
    else:
        alpha /= tau
        f_alpha, x_alpha = f(alpha)
        while f0 <= f_alpha and alpha > MIN_STEP_SIZE:
            alpha /= tau
            f_alpha, x_alpha = f(alpha)
        return f_alpha, x_alpha, alpha
