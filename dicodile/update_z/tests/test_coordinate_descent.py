import numpy as np

from dicodile.utils.csc import compute_objective
from dicodile.update_z.coordinate_descent import _init_beta


def test_init_beta():
    n_atoms = 5
    n_channels = 2
    height, width = 31, 37
    height_atom, width_atom = 11, 13
    height_valid = height - height_atom + 1
    width_valid = width - width_atom + 1

    rng = np.random.RandomState(42)

    X = rng.randn(n_channels, height, width)
    D = rng.randn(n_atoms, n_channels, height_atom, width_atom)
    D /= np.sqrt(np.sum(D * D, axis=(1, 2, 3), keepdims=True))
    # z = np.zeros((n_atoms, height_valid, width_valid))
    z = rng.randn(n_atoms, height_valid, width_valid)

    lmbd = 1
    beta, dz_opt, dE = _init_beta(X, D, lmbd, z_i=z)

    assert beta.shape == z.shape
    assert dz_opt.shape == z.shape

    for _ in range(50):
        k = rng.randint(n_atoms)
        h = rng.randint(height_valid)
        w = rng.randint(width_valid)

        # Check that the optimal value is independent of the current value
        z_old = z[k, h, w]
        z[k, h, w] = rng.randn()
        beta_new, *_ = _init_beta(X, D, lmbd, z_i=z)
        assert np.isclose(beta_new[k, h, w], beta[k, h, w])

        # Check that the chosen value is optimal
        z[k, h, w] = z_old + dz_opt[k, h, w]
        c0 = compute_objective(X, z, D, lmbd)

        eps = 1e-5
        z[k, h, w] -= 3.5 * eps
        for _ in range(5):
            z[k, h, w] += eps
            assert c0 <= compute_objective(X, z, D, lmbd)
        z[k, h, w] = z_old
