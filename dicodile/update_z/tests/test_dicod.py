import pytest
import numpy as np


from dicodile.update_z.dicod import dicod
from dicodile.utils import check_random_state
from dicodile.utils.csc import compute_ztz, compute_ztX
from dicodile.utils.shape_helpers import get_full_support
from dicodile.update_z.coordinate_descent import _init_beta
from dicodile.utils.csc import reconstruct, compute_objective

VERBOSE = 100
N_WORKERS = 4


@pytest.mark.parametrize('signal_support, atom_support',
                         [((800,), (50,)), ((100, 100), (10, 8))])
@pytest.mark.parametrize('n_workers', [2, 6, N_WORKERS])
def test_stopping_criterion(n_workers, signal_support, atom_support):
    tol = 1
    reg = 1
    n_atoms = 10
    n_channels = 3

    rng = check_random_state(42)

    X = rng.randn(n_channels, *signal_support)
    D = rng.randn(n_atoms, n_channels, *atom_support)
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))

    z_hat, *_ = dicod(X, D, reg, tol=tol, n_workers=n_workers, verbose=VERBOSE)

    beta, dz_opt, _ = _init_beta(X, D, reg, z_i=z_hat)
    assert abs(dz_opt).max() < tol


@pytest.mark.parametrize('valid_support, atom_support', [((500,), (30,)),
                                                         ((72, 60), (10, 8))])
def test_ztz(valid_support, atom_support):
    tol = .5
    reg = .1
    n_atoms = 7
    n_channels = 5
    random_state = None

    sig_support = get_full_support(valid_support, atom_support)

    rng = check_random_state(random_state)

    X = rng.randn(n_channels, *sig_support)
    D = rng.randn(n_atoms, n_channels, *atom_support)
    D /= np.sqrt(np.sum(D * D, axis=(1, 2), keepdims=True))

    z_hat, ztz, ztX, *_ = dicod(X, D, reg, tol=tol, n_workers=N_WORKERS,
                                return_ztz=True, verbose=VERBOSE)

    ztz_full = compute_ztz(z_hat, atom_support)
    assert np.allclose(ztz_full, ztz)

    ztX_full = compute_ztX(z_hat, X)
    assert np.allclose(ztX_full, ztX)


@pytest.mark.parametrize('valid_support, atom_support, reg',
                         [((500,), (30,), 1), ((72, 60), (10, 8), 100)])
def test_warm_start(valid_support, atom_support, reg):
    tol = 1
    n_atoms = 7
    n_channels = 5
    random_state = 36

    rng = check_random_state(random_state)

    D = rng.randn(n_atoms, n_channels, *atom_support)
    D /= np.sqrt(np.sum(D * D, axis=(1, 2), keepdims=True))
    z = rng.randn(n_atoms, *valid_support)
    z *= (rng.rand(n_atoms, *valid_support) > .7)

    X = reconstruct(z, D)

    z_hat, *_ = dicod(X, D, reg=0, z0=z, tol=tol, n_workers=N_WORKERS,
                      max_iter=10000, verbose=VERBOSE)
    assert np.allclose(z_hat, z)

    X = rng.randn(*X.shape)

    z_hat, *_ = dicod(X, D, reg, z0=z, tol=tol, n_workers=N_WORKERS,
                      max_iter=100000, verbose=VERBOSE)
    beta, dz_opt, _ = _init_beta(X, D, reg, z_i=z_hat)
    assert np.all(dz_opt <= tol)


@pytest.mark.parametrize('valid_support, atom_support', [((500,), (30,)),
                                                         ((72, 60), (10, 8))])
def test_freeze_support(valid_support, atom_support):
    tol = .5
    reg = 0
    n_atoms = 7
    n_channels = 5
    random_state = None

    sig_support = get_full_support(valid_support, atom_support)

    rng = check_random_state(random_state)

    D = rng.randn(n_atoms, n_channels, *atom_support)
    D /= np.sqrt(np.sum(D * D, axis=(1, 2), keepdims=True))
    z = rng.randn(n_atoms, *valid_support)
    z *= rng.rand(n_atoms, *valid_support) > .5

    X = rng.randn(n_channels, *sig_support)

    z_hat, *_ = dicod(X, D, reg, z0=0 * z, tol=tol, n_workers=N_WORKERS,
                      max_iter=1000, freeze_support=True, verbose=VERBOSE)
    assert np.all(z_hat == 0)

    z_hat, *_ = dicod(X, D, reg, z0=z, tol=tol, n_workers=N_WORKERS,
                      max_iter=1000, freeze_support=True, verbose=VERBOSE)

    assert np.all(z_hat[z == 0] == 0)


@pytest.mark.parametrize('valid_support, atom_support', [((500,), (30,)),
                                                         ((72, 60), (10, 8))])
def test_cost(valid_support, atom_support):

    tol = .5
    reg = 0
    n_atoms = 7
    n_channels = 5
    random_state = None

    sig_support = get_full_support(valid_support, atom_support)

    rng = check_random_state(random_state)

    D = rng.randn(n_atoms, n_channels, *atom_support)
    D /= np.sqrt(np.sum(D * D, axis=(1, 2), keepdims=True))
    z = rng.randn(n_atoms, *valid_support)
    z *= rng.rand(n_atoms, *valid_support) > .5

    X = rng.randn(n_channels, *sig_support)

    z_hat, *_, pobj, _ = dicod(X, D, reg, z0=z, tol=tol, n_workers=N_WORKERS,
                               max_iter=1000, freeze_support=True,
                               verbose=VERBOSE)
    cost = pobj[-1][2]
    assert np.isclose(cost, compute_objective(X, z_hat, D, reg))
