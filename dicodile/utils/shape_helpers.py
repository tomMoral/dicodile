import numpy as np
from numba import njit


def get_full_support(valid_support, atom_support):
    return tuple([
        size_valid_ax + size_atom_ax - 1
        for size_valid_ax, size_atom_ax in zip(valid_support, atom_support)
    ])


def get_valid_support(sig_support, atom_support):
    return tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_support, atom_support)
    ])


# TODO: improve to find something that fits better the constraints
def find_grid_size(n_workers, sig_support, atom_support):
    """Given a signal support and a number of jobs, find a suitable grid shape

    If the signal has a 1D support, (n_workers,) is returned.

    If the signal has a 2D support, the grid size is computed such that the
    area of the signal support in each worker is the most balance.

    Parameters
    ----------
    n_workers: int
        Number of workers available
    sig_support: tuple
        Size of the support of the signal to decompose.
    atom_support: tuple
        Size of the support of the atoms to learn.
    """
    if len(sig_support) == 1:
        return (n_workers,)
    elif len(sig_support) == 2:
        width, height = sig_support
        w_atom, h_atom = atom_support
        max_w = max(1, width // (2*w_atom))
        max_h = max(1, height // (2*h_atom))

        w_world, h_world = 1, n_workers
        w_ratio = width * n_workers / height
        if n_workers > max_h:
            w_ratio = np.inf

        for w in range(2, max_w + 1):
            if n_workers % w != 0:
                continue
            h = n_workers // w
            ratio = width / w * (h / height)
            if (abs(ratio - 1) < abs(w_ratio - 1) and h <= max_h):
                w_ratio = ratio
                w_world, h_world = w, h
        assert w_ratio < np.inf, (
            f"could not find suitable topology for {n_workers} workers. "
            f"The signal size is {sig_support} and the atom size is "
            f"{atom_support}"
        )
        return w_world, h_world
    else:
        raise NotImplementedError("")


@njit(cache=True)
def fast_unravel(i, shape):
    pt = []
    for v in shape[::-1]:
        pt.insert(0, i % v)
        i //= v
    return pt


@njit(cache=True)
def fast_unravel_offset(i, shape, offset):
    pt = []
    for v, offset_axis in zip(shape[::-1], offset[::-1]):
        pt.insert(0, i % v + offset_axis)
        i //= v
    return pt
