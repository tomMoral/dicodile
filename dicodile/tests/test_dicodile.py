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
