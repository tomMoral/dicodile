import itertools
import numpy as np


from . import check_random_state
from .shape_helpers import fast_unravel, fast_unravel_offset


def get_coordinate_iterator(shape, strategy, random_state):
    order = np.array(list(itertools.product(*[range(v) for v in shape])))
    order_ = order.copy()
    n_coordinates = np.prod(shape)

    rng = check_random_state(random_state)

    def iter_coord():
        i = 0
        if strategy == 'random':
            def shuffle():
                order[:] = order_[rng.choice(range(n_coordinates),
                                             size=n_coordinates)]
        elif strategy == 'cyclic-r':
            def shuffle():
                order[:] = order[rng.choice(range(n_coordinates),
                                            size=n_coordinates, replace=False)]
        else:
            def shuffle():
                pass

        while True:
            j = i % n_coordinates
            if j == 0:
                shuffle()
            yield order[j]
            i += 1

    return iter_coord()


def get_order_iterator(shape, strategy, random_state, offset=None):

    rng = check_random_state(random_state)
    n_coordinates = np.prod(shape)

    if offset is None:
        def unravel(i0):
            return tuple(fast_unravel(i0, shape))
    else:
        def unravel(i0):
            return tuple(fast_unravel_offset(i0, shape, offset))

    if strategy == 'cyclic':
        # return itertools.cycle(range(n_coordinates))
        order = np.arange(n_coordinates)

        def shuffle():
            pass
    else:
        replace = strategy == 'random'
        order = rng.choice(range(n_coordinates), size=n_coordinates,
                           replace=replace)

        def shuffle():
            order[:] = rng.choice(n_coordinates, size=n_coordinates,
                                  replace=replace)

    def iter_order():
        while True:
            shuffle()
            for i0 in order:
                yield unravel(i0)

    return iter_order()
