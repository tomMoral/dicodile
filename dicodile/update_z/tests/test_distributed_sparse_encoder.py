import pytest
import numpy as np

from dicodile.utils import check_random_state
from dicodile.utils.dictionary import compute_DtD

from dicodile.update_z.distributed_sparse_encoder import\
    DistributedSparseEncoder


def test_distributed_sparse_encoder():
    rng = check_random_state(42)

    n_atoms = 10
    n_channels = 3
    n_times_atom = 10
    n_times = 10 * n_times_atom
    reg = 5e-1

    params = dict(tol=1e-2, n_seg='auto', timing=False, timeout=None,
                  verbose=10, strategy='greedy', max_iter=100000,
                  soft_lock='border', z_positive=True, return_ztz=False,
                  freeze_support=False, random_state=27)

    X = rng.randn(n_channels, n_times)
    D = rng.randn(n_atoms, n_channels, n_times_atom)
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
    DtD = compute_DtD(D)

    encoder = DistributedSparseEncoder(n_workers=2)

    encoder.init_workers(X, D, reg, params, DtD=DtD)

    with pytest.raises(ValueError, match=r"pre-computed value DtD"):
        encoder.set_worker_D(D)
