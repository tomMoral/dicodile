import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory


from dicodile.update_z.dicod import dicod
from dicodile.data.images import get_mandril
from dicodile.utils.segmentation import Segmentation
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.dictionary import init_dictionary
from dicodile.utils.shape_helpers import get_valid_support
from dicodile.utils.csc import compute_objective, reconstruct

mem = Memory(location='.')


@mem.cache
def run(n_atoms=25, atom_support=(12, 12), reg=.01,
                          tol=5e-2, n_workers=100, random_state=60):
    rng = np.random.RandomState(random_state)

    X = get_mandril()
    D_init = init_dictionary(X, n_atoms, atom_support, random_state=rng)
    lmbd_max = get_lambda_max(X, D_init).max()
    reg_ = reg * lmbd_max

    z_hat, *_ = dicod(
        X, D_init, reg_, max_iter=1000000, n_workers=n_workers, tol=tol,
        strategy='greedy', verbose=1, soft_lock='border', z_positive=False,
        timing=False)
    pobj = compute_objective(X, z_hat, D_init, reg_)
    z_hat = np.clip(z_hat, -1e3, 1e3)
    print("[DICOD] final cost : {}".format(pobj))

    X_hat = reconstruct(z_hat, D_init)
    X_hat = np.clip(X_hat, 0, 1)
    return X_hat, pobj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--no-cache', action='store_true',
                        help='Re-run the entire computations')
    args = parser.parse_args()

    # Args
    reg = .01
    tol = 5e-2
    n_atoms = 25
    w_world = 7
    n_workers = w_world * w_world
    random_state = 60
    atom_support = (16, 16)

    run_args = (n_atoms, atom_support, reg, tol, n_workers, random_state)
    if args.no_cache:
        X_hat, pobj = run.call(*run_args)[0]
    else:
        X_hat, pobj = run(*run_args)

    file_name = f"mandril_M{n_workers}_support{atom_support[0]}"
    np.save(f"benchmarks_results/{file_name}_X_hat.npy", X_hat)

    # Compute the worker segmentation for the image,
    n_channels, *sig_support = X_hat.shape
    valid_support = get_valid_support(sig_support, atom_support)
    workers_segments = Segmentation(n_seg=(w_world, w_world),
                                    signal_support=valid_support,
                                    overlap=0)

    fig = plt.figure("recovery")
    fig.patch.set_alpha(0)

    ax = plt.subplot()
    ax.imshow(X_hat.swapaxes(0, 2))
    for i_seg in range(workers_segments.effective_n_seg):
        seg_bounds = np.array(workers_segments.get_seg_bounds(i_seg))
        seg_bounds = seg_bounds + np.array(atom_support) / 2
        ax.vlines(seg_bounds[1], *seg_bounds[0], linestyle='--')
        ax.hlines(seg_bounds[0], *seg_bounds[1], linestyle='--')
    ax.axis('off')
    plt.tight_layout()

    fig.savefig(f"benchmarks_results/{file_name}.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0)
    print("done")
