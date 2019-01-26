import numpy as np


NEIGHBOR_POS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)]


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def get_neighbors(i, grid_shape):
    """Return a list of existing neighbors for a given cell in a grid

    Parameters
    ----------
    i : int
        index of the cell in the grid
    grid_shape : 2-tuple
        Size of the considered grid.

    Return
    ------
    neighbors : list
        List with 8 elements. Return None if the neighbor does not exist and
        the ravel indice of the neighbor if it exists.
    """
    height, width = grid_shape
    assert 0 <= i < height * width
    h_cell, w_cell = i // height, i % height

    neighbors = [None] * 8
    for i, (dh, dw) in enumerate(NEIGHBOR_POS):
        h_neighbor = h_cell + dh
        w_neighbor = w_cell + dw
        has_neighbor = 0 <= h_neighbor < height
        has_neighbor &= 0 <= w_neighbor < width
        if has_neighbor:
            neighbors[i] = h_neighbor * width + w_neighbor

    return neighbors
