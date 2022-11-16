import pytest
import numpy as np
from scipy.signal import fftconvolve

from dicodile.utils import check_random_state
from dicodile.utils.csc import reconstruct, compute_ztz
from dicodile.utils.csc import _dense_convolve_multi_uv
from dicodile.utils.csc import _dense_transpose_convolve
from dicodile.utils.dictionary import compute_DtD, get_D
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


@pytest.mark.parametrize('atom_support',
                         [(35, ), (40, 30), (10, 12, 3)])
def test_dense_convolve_multi_uv_shape(atom_support):

    n_channels = 3
    sig_shape = (n_channels, *[8 * n for n in atom_support])
    n_atoms = 25
    valid_support = get_valid_support(sig_support=sig_shape[1:],
                                      atom_support=atom_support)

    z_hat = np.ones((n_atoms, *valid_support))
    u = np.ones((n_atoms, n_channels))
    v = np.ones((n_atoms, *atom_support))
    Xi = _dense_convolve_multi_uv(z_hat, (u, v))

    assert Xi.shape == sig_shape


def test_convolve_uv_and_convolve_d_match():
    n_channels = 3
    sig_shape = (n_channels, 800, 600)
    atom_shape = (n_channels, 40, 30)
    atom_support = atom_shape[1:]
    n_atoms = 20
    valid_support = get_valid_support(sig_support=sig_shape[1:],
                                      atom_support=atom_support)
    rng = np.random.default_rng(seed=42)
    z_hat = rng.uniform(size=(n_atoms, *valid_support))

    u = rng.uniform(size=(n_atoms, n_channels))
    v = rng.uniform(size=(n_atoms, *atom_support))
    uv_convolution = reconstruct(z_hat, (u, v))

    d = get_D(u, v)
    d_convolution = reconstruct(z_hat, d)

    assert np.allclose(uv_convolution, d_convolution)


@pytest.mark.parametrize('atom_support',
                         [(35, ), (40, 30), (10, 12, 3)])
def test_rank1_DtD_matches_full_DtD(atom_support):
    n_channels = 3
    n_atoms = 25

    rng = np.random.default_rng(seed=42)
    u = rng.uniform(size=(n_atoms, n_channels))
    v = rng.uniform(size=(n_atoms, *atom_support))
    D = get_D(u, v)

    d_dtd = compute_DtD(D)
    uv_dtd = compute_DtD((u, v))

    assert np.allclose(d_dtd, uv_dtd)


def test_dense_transpose_convolve_uv_and_D_match():
    n_channels = 3
    n_atoms = 25
    atom_support = (40, 32)
    signal_support = (800, 600)
    rng = np.random.default_rng(seed=42)
    u = rng.uniform(size=(n_atoms, n_channels))
    v = rng.uniform(size=(n_atoms, *atom_support))
    D = get_D(u, v)
    residuals_i = np.ones((n_channels, *signal_support))

    dtc_d = _dense_transpose_convolve(residuals_i, D)
    dtc_uv = _dense_transpose_convolve(residuals_i, (u, v))

    assert np.allclose(dtc_uv, dtc_d)
