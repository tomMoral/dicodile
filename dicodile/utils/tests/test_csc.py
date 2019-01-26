import pytest
import numpy as np
from scipy.signal import fftconvolve

from dicodile.utils.csc import reconstruct
from dicodile.utils.csc import compute_ztz
from dicodile.utils import check_random_state


@pytest.mark.parametrize('valid_shape, atom_shape', [((500,), (30,)),
                                                     ((72, 60), (10, 8))])
@pytest.mark.parametrize('sparsity', [1, .01])
def test_ztz(valid_shape, atom_shape, sparsity):
    n_atoms = 7
    n_channels = 5
    random_state = None

    rng = check_random_state(random_state)

    z = rng.randn(n_atoms, *valid_shape)
    z *= rng.rand(*z.shape) < sparsity
    D = rng.randn(n_atoms, n_channels, *atom_shape)

    ztz = compute_ztz(z, atom_shape)
    grad = np.sum([[[fftconvolve(ztz_k0_k, d_kp, mode='valid') for d_kp in d_k]
                    for ztz_k0_k, d_k in zip(ztz_k0, D)]
                   for ztz_k0 in ztz], axis=1)
    cost = np.dot(D.ravel(), grad.ravel())

    X_hat = reconstruct(z, D)

    assert np.isclose(cost, np.dot(X_hat.ravel(), X_hat.ravel()))
