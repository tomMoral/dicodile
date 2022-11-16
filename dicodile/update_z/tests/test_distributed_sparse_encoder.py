import pytest
import numpy as np

from dicodile.utils import check_random_state
from dicodile.utils.dictionary import compute_DtD, get_D, get_max_error_patch
from dicodile.utils.csc import compute_objective
from dicodile.utils.csc import compute_ztX, compute_ztz

from dicodile.update_z.distributed_sparse_encoder import\
    DistributedSparseEncoder


def _prox_d(D):
    sum_axis = tuple(range(1, D.ndim))
    norm_d = np.maximum(1, np.linalg.norm(D, axis=sum_axis, keepdims=True))
    return D / norm_d


@pytest.mark.parametrize('rank1', [True, False])
def test_distributed_sparse_encoder(rank1):
    rng = check_random_state(42)

    n_atoms = 10
    n_channels = 3
    atom_support = (10,)
    n_times = 10 * atom_support[0]
    reg = 5e-1

    params = dict(tol=1e-2, n_seg='auto', timing=False, timeout=None,
                  verbose=100, strategy='greedy', max_iter=100000,
                  soft_lock='border', z_positive=True, return_ztz=False,
                  freeze_support=False, warm_start=False, random_state=27)

    X = rng.randn(n_channels, n_times)
    if not rank1:
        D = rng.randn(n_atoms, n_channels, *atom_support)
        sum_axis = tuple(range(1, D.ndim))
        D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
    else:
        u = _prox_d(rng.randn(n_atoms, n_channels))
        v = _prox_d(rng.randn(n_atoms, *atom_support))

        D = u, v

    DtD = compute_DtD(D)
    encoder = DistributedSparseEncoder(n_workers=2)
    encoder.init_workers(X, D, reg, params, DtD=DtD)

    encoder.process_z_hat()
    z_hat = encoder.get_z_hat()

    # Check that distributed computations are correct for cost and sufficient
    # statistics
    cost_distrib = encoder.get_cost()
    if rank1:
        u, v = D
        D = get_D(u, v)
    cost = compute_objective(X, z_hat, D, reg)
    assert np.allclose(cost, cost_distrib)

    ztz_distrib, ztX_distrib = encoder.get_sufficient_statistics()
    ztz = compute_ztz(z_hat, atom_support)
    ztX = compute_ztX(z_hat, X)
    assert np.allclose(ztz, ztz_distrib)
    assert np.allclose(ztX, ztX_distrib)

    encoder.shutdown_workers()


def test_pre_computed_DtD_should_always_be_passed_to_set_worker_D():
    rng = check_random_state(42)

    n_atoms = 10
    n_channels = 3
    atom_support = (10,)
    n_times = 10 * atom_support[0]
    reg = 5e-1

    params = dict(tol=1e-2, n_seg='auto', timing=False, timeout=None,
                  verbose=100, strategy='greedy', max_iter=100000,
                  soft_lock='border', z_positive=True, return_ztz=False,
                  freeze_support=False, warm_start=False, random_state=27)

    X = rng.randn(n_channels, n_times)
    D = rng.randn(n_atoms, n_channels, *atom_support)
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))

    DtD = compute_DtD(D)
    encoder = DistributedSparseEncoder(n_workers=2)
    encoder.init_workers(X, D, reg, params, DtD=DtD)

    with pytest.raises(ValueError, match=r"pre-computed value DtD"):
        encoder.set_worker_D(D)


@pytest.mark.parametrize("n_workers", [1, 2, 3])
def test_compute_max_error_patch(n_workers):
    rng = check_random_state(42)

    n_atoms = 2
    n_channels = 3
    n_times_atom = 10
    n_times = 10 * n_times_atom
    reg = 5e-1

    params = dict(tol=1e-2, n_seg='auto', timing=False, timeout=None,
                  verbose=100, strategy='greedy', max_iter=100000,
                  soft_lock='border', z_positive=True, return_ztz=False,
                  freeze_support=False, warm_start=False, random_state=27)

    X = rng.randn(n_channels, n_times)
    D = rng.randn(n_atoms, n_channels, n_times_atom)
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))

    encoder = DistributedSparseEncoder(n_workers=n_workers)

    encoder.init_workers(X, D, reg, params, DtD=None)

    encoder.process_z_hat()
    z_hat = encoder.get_z_hat()

    max_error_patch = encoder.compute_and_get_max_error_patch()
    assert max_error_patch.shape == (n_channels, n_times_atom)

    reference_patch, _ = get_max_error_patch(X, z_hat, D)
    assert np.allclose(max_error_patch, reference_patch)

    encoder.shutdown_workers()


@pytest.mark.parametrize('rank1', [True, False])
@pytest.mark.parametrize('warm_start', [True, False])
def test_grow_n_atoms(rank1, warm_start):
    rng = check_random_state(42)

    n_channels = 3
    atom_support = (10,)
    n_times = 10 * atom_support[0]
    reg = 5e-1

    params = dict(tol=1e-2, n_seg='auto', timing=False, timeout=None,
                  verbose=100, strategy='greedy', max_iter=100000,
                  soft_lock='border', z_positive=True, return_ztz=False,
                  freeze_support=False, warm_start=warm_start, random_state=27)

    X = rng.randn(n_channels, n_times)

    def make_dict(n_atoms):
        if not rank1:
            D = rng.randn(n_atoms, n_channels, *atom_support)
            sum_axis = tuple(range(1, D.ndim))
            D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
        else:
            u = _prox_d(rng.randn(n_atoms, n_channels))
            v = _prox_d(rng.randn(n_atoms, *atom_support))

            D = u, v
        return D

    # Init dict with zero atom
    D = make_dict(0)

    encoder = DistributedSparseEncoder(n_workers=3)
    encoder.init_workers(X, D, reg, params, DtD=None)

    # update dict with one atom
    D = make_dict(1)
    encoder.set_worker_D(D)

    # Process z hat
    encoder.process_z_hat()

    # Add one atom
    # not quite what we would do in practice
    # (we would add atoms and not make a new dict)
    D = make_dict(2)
    encoder.set_worker_D(D)

    # Process z hat
    encoder.process_z_hat()
    z_hat = encoder.get_z_hat()

    # Check z_hat shape
    assert z_hat.shape[0] == 2


@pytest.mark.parametrize('rank1', [True, False])
def test_cannot_shrink_n_atoms(rank1):
    rng = check_random_state(42)

    n_channels = 3
    atom_support = (10,)
    n_times = 10 * atom_support[0]
    reg = 5e-1

    params = dict(tol=1e-2, n_seg='auto', timing=False, timeout=None,
                  verbose=100, strategy='greedy', max_iter=100000,
                  soft_lock='border', z_positive=True, return_ztz=False,
                  freeze_support=False, warm_start=False, random_state=27)

    X = rng.randn(n_channels, n_times)

    def make_dict(n_atoms):
        if not rank1:
            D = rng.randn(n_atoms, n_channels, *atom_support)
            sum_axis = tuple(range(1, D.ndim))
            D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
        else:
            u = _prox_d(rng.randn(n_atoms, n_channels))
            v = _prox_d(rng.randn(n_atoms, *atom_support))

            D = u, v
        return D

    D = make_dict(2)

    encoder = DistributedSparseEncoder(n_workers=3)
    encoder.init_workers(X, D, reg, params, DtD=None)

    # Process z hat
    encoder.process_z_hat()

    # remove one atom
    D = make_dict(1)

    with pytest.raises(AssertionError):
        encoder.set_worker_D(D)
