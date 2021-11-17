import pytest
import numpy as np
from scipy.signal import fftconvolve

from dicodile.utils.csc import reconstruct
from dicodile.utils.csc import compute_ztz
from dicodile.utils.csc import _dense_convolve_multi_uv
from dicodile.utils import check_random_state
from dicodile.utils.shape_helpers import get_valid_support


@pytest.mark.parametrize('valid_support, atom_support', [((500,), (30,)),
                                                         ((72, 60), (10, 8))])
@pytest.mark.parametrize('sparsity', [1, .01])
def test_ztz(valid_support, atom_support, sparsity):
    n_atoms = 7
    n_channels = 5
    random_state = None

    rng = check_random_state(random_state)

    z = rng.randn(n_atoms, *valid_support)
    z *= rng.rand(*z.shape) < sparsity
    D = rng.randn(n_atoms, n_channels, *atom_support)

    ztz = compute_ztz(z, atom_support)
    grad = np.sum([[[fftconvolve(ztz_k0_k, d_kp, mode='valid') for d_kp in d_k]
                    for ztz_k0_k, d_k in zip(ztz_k0, D)]
                   for ztz_k0 in ztz], axis=1)
    cost = np.dot(D.ravel(), grad.ravel())

    X_hat = reconstruct(z, D)

    assert np.isclose(cost, np.dot(X_hat.ravel(), X_hat.ravel()))


@pytest.mark.parametrize('atom_single_channel_shape',
                         [(35, ), (40, 30), (10, 12, 3)])
def test_dense_convolve_multi_uv_shape(atom_single_channel_shape):

    n_channels = 3
    sig_shape = (n_channels, *[8 * n for n in atom_single_channel_shape])
    atom_shape = (n_channels, *atom_single_channel_shape)
    atom_support = atom_shape[1:]
    n_atoms = 25
    valid_support = get_valid_support(sig_support=sig_shape[1:],
                                      atom_support=atom_support)

    z_hat = np.ones((n_atoms, *valid_support))
    u = np.ones((n_atoms, n_channels))
    v = np.ones((n_atoms, *atom_support))
    Xi = _dense_convolve_multi_uv(z_hat, (u, v))

    assert Xi.shape == sig_shape
