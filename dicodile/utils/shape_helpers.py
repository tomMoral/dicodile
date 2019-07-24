

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


def find_grid_size(n_jobs, sig_support):
    """Given a signal support and a number of jobs, find asuitable grid shape

    If the signal has a 1D support, (n_jobs,) is returned.

    If the signal has a 2D support, the grid size is computed such that the
    area of the signal support in each worker is the most balance.

    Parameters
    ----------
    n_jobs: int
        Number of workers available
    sig_support: tuple
        Size of the support of the signal to decompose.
    """
    if len(sig_support) == 1:
        return (n_jobs,)
    elif len(sig_support) == 2:
        height, width = sig_support
        w_world, h_world = 1, n_jobs
        w_ratio = width * n_jobs / height
        for i in range(2, n_jobs + 1):
            if n_jobs % i != 0:
                continue
            j = n_jobs // i
            ratio = width * j / (height * i)
            if abs(ratio - 1) < abs(w_ratio - 1):
                w_ratio = ratio
                w_world, h_world = i, j
        return w_world, h_world
    else:
        raise NotImplementedError("")
