import numpy as np

from ..utils.dictionary import get_lambda_max


def simulate_data(n_times, n_times_atom, n_atoms, n_channels, noise_level,
                  random_state=None):
    rng = np.random.RandomState(random_state)
    rho = n_atoms / (n_channels * n_times_atom)
    D = rng.normal(scale=10.0, size=(n_atoms, n_channels, n_times_atom))
    D = np.array(D)
    nD = np.sqrt((D * D).sum(axis=-1, keepdims=True))
    D /= nD + (nD == 0)

    Z = (rng.rand(n_atoms, (n_times - 1) * n_times_atom + 1) < rho
         ).astype(np.float64)
    Z *= rng.normal(scale=10, size=(n_atoms, (n_times - 1) * n_times_atom + 1))

    X = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, Z)]).sum(axis=0)
    X += noise_level * rng.normal(size=X.shape)

    lmbd_max = get_lambda_max(X, D)

    return X, D, lmbd_max
