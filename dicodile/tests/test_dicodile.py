import pytest

import numpy as np

from dicodile import dicodile
from dicodile.data.simulate import simulate_data

from dicodile.utils.testing import is_deacreasing


def test_dicodile():

    X, D, _ = simulate_data(n_times=100, n_times_atom=10, n_atoms=2,
                            n_channels=3, noise_level=1e-5, random_state=42)

    D_hat, z_hat, pobj, times = dicodile(
        X, D, reg=.1, z_positive=True, n_iter=10, eps=1e-4,
        n_workers=1, verbose=2, tol=1e-10)
    assert is_deacreasing(pobj)


@pytest.mark.parametrize("n_workers", [1, 2, 3])
def test_dicodile_greedy(n_workers):
    n_channels = 3
    n_atoms = 2
    n_times_atom = 10

    X, D, _ = simulate_data(n_times=100, n_times_atom=n_times_atom,
                            n_atoms=n_atoms, n_channels=n_channels,
                            noise_level=1e-5, random_state=42)

    # Starts with a single random atom, expect to learn others
    # from the largest reconstruction error patch
    D[1:] = np.zeros((n_atoms-1, n_channels, n_times_atom))

    D_hat, z_hat, pobj, times = dicodile(
        X, D, reg=.1, z_positive=True, n_iter=10, eps=1e-4,
        n_workers=n_workers, verbose=2, tol=1e-10)

    assert is_deacreasing(pobj)
