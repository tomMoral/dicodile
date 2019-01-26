
import numpy as np
from scipy import sparse

from dicodile.data import get_hubble
from dicodile.dicodile import dicodile
from dicodile.utils.viz import plot_atom_and_coefs

from dicodile.utils.dictionary import init_dictionary


n_atoms = 25
random_state = 42


def run_dicodile_hubble(size, reg, L):
    X = get_hubble(size=size)

    D_init = init_dictionary(
        X, n_atoms, (L, L), random_state=random_state)

    dicod_kwargs = dict(soft_lock='border')
    pobj, times, D_hat, z_hat = dicodile(
        X, D_init, reg=reg, z_positive=True, n_iter=100, n_jobs=400,
        eps=1e-5, tol=1e-3, verbose=2, dicod_kwargs=dicod_kwargs)

    # Save the atoms
    prefix = (f"K{n_atoms}_L{L}_reg{reg}"
              f"_seed{random_state}_dicodile_{size}_")
    prefix = prefix.replace(" ", "")
    np.save(f"hubble/{prefix}D_hat.npy", D_hat)
    z_hat[z_hat < 1e-2] = 0
    z_hat_save = [sparse.csr_matrix(z) for z in z_hat]
    np.save(f"hubble/{prefix}z_hat.npy", z_hat_save)

    plot_atom_and_coefs(D_hat, z_hat, prefix)


def plot_dicodile_hubble(size, reg, L):
    # Save the atoms
    prefix = (f"K{n_atoms}_L{L}_reg{reg}"
              f"_seed{random_state}_dicodile_{size}_")
    D_hat = np.load(f"hubble/{prefix}D_hat.npy")
    z_hat = np.load(f"hubble/{prefix}z_hat.npy")
    plot_atom_and_coefs(D_hat, z_hat, prefix)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results from saved dictionaries')
    parser.add_argument('--all', action='store_true',
                        help='Plot the results from saved dictionaries')
    args = parser.parse_args()

    display_params = ("Medium", .1, 32)

    if args.plot:
        run_func = plot_dicodile_hubble
    else:
        run_func = run_dicodile_hubble

    if args.all:
        for size in ['Large', 'Medium']:

            for reg in [.1, .3, .05]:
                for L in [32, 28]:
                    try:
                        run_func(size, ref, L)
                    except FileNotFoundError:
                        continue
    else:
        run_func(*display_params)
