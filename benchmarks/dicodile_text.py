from datetime import datetime
import numpy as np
import os
import pandas as pd
import pathlib
from scipy import signal
from scipy.optimize import linear_sum_assignment

# Import for online DL
import spams
from sklearn.feature_extraction.image import extract_patches_2d

# Import for CDL
from dicodile import dicodile
from dicodile.config import DATA_HOME

# Import to initiate the dictionary
from dicodile.update_d.update_d import tukey_window
from dicodile.utils import check_random_state
from dicodile.utils.dictionary import init_dictionary, prox_d


# Caching utility
from joblib import Memory

memory = Memory(location='.', verbose=0)


BASE_FILE_NAME = os.path.basename(__file__)
OUTPUT_DIR = pathlib.Path('benchmarks_results')
DATA_DIR = DATA_HOME / 'images' / 'text'


@memory.cache
def compute_dl(X, n_atoms, atom_support, reg=.2,
               max_patches=1_000_000, n_jobs=10):
    """Compute dictionary using Online dictionary learning.

    Parameters
    ----------
    X : ndarray, shape n_channels, *signal_support)
        Signal from which the patterns are extracted. Note that this
        function is only working for a single image and a single channel.
    n_atoms : int
        Number of pattern to learn form the data
    atom_support : tuple(int, int)
        Support of the patterns that are learned.
    max_patches: int (default: 1_000_000)
        Maximal number of patches extracted from the image to learn
        the dictionary. Taking this parameter too large might result in
        memory overflow.
    n_jobs: int (default: 10)
        Number of CPUs that can be used for the computations.

    Returns
    -------
    D_hat: ndarray, shape (n_atoms, n_channels, *atom_support)
        The learned dictionary
    """
    n_channels, *_ = X.shape
    assert n_channels == 1, (
        f'n_channels larger than 1 is not supported. Got {n_channels}'
    )
    X = X[0]

    # extract 2d patches from the image
    X_dl = extract_patches_2d(X, atom_support, max_patches=max_patches)
    X_dl = X_dl.reshape(X_dl.shape[0], -1)
    norm = np.linalg.norm(X_dl, axis=1)
    mask = norm != 0
    X_dl = X_dl[mask]
    X_dl /= norm[mask][:, None]

    meta = dict(lambda1=reg, iter=10_000, mode=2, posAlpha=True, posD=False)

    # Learn the dictionary with spams
    D_dl = spams.trainDL(np.asfortranarray(X_dl.T, dtype=np.float),
                         numThreads=n_jobs, batchsize=512,
                         K=n_atoms, **meta, verbose=False).T

    return D_dl.reshape(n_atoms, 1, *atom_support), meta


@memory.cache
def compute_cdl(X, n_atoms, atom_support, D_init, reg=.2,
                window=False, n_jobs=10):
    """Compute dictionary using Dicodile.

    Parameters
    ----------
    X : ndarray, shape (n_channels, *signal_support)
        Signal from which the patterns are extracted. Note that this
        function is only working for a single image and a single channel.
    n_atoms : int
        Number of pattern to learn form the data
    atom_support : tuple(int, int)
        Support of the patterns that are learned.
    D_init: ndarray, shape (n_atoms, n_channels, *atom_support)
        Initial dictionary, used to start the algorithm.
    window: boolean (default: False)
        If set to True, use a window to force dictionary boundaries to zero.
    n_jobs: int (default: 10)
        Number of CPUs that can be used for the computations.

    Returns
    -------
    D_hat: ndarray, shape (n_atoms, n_channels, *atom_support)
        The learned dictionary
    """

    # Add a small noise to avoid having coefficients that are equals. They
    # might make the distributed optimization complicated.
    X_0 = X.copy()
    X_0 += X_0.std() * 1e-8 * np.random.randn(*X.shape)

    meta = dict(reg=reg, tol=1e-3, z_positive=True, n_iter=100,
                window=window)

    # fit the dictionary with dicodile
    D_hat, z_hat, pobj, times = dicodile(
        X_0, D_init, n_workers=n_jobs, w_world='auto',
        **meta, verbose=1,
    )

    # Order the dictionary based on the l1 norm of its activation
    i0 = abs(z_hat).sum(axis=(1, 2)).argsort()[::-1]
    return D_hat[i0], meta


def get_input(filename):
    data = np.load(DATA_DIR / filename)
    X = data.get('X')[None]
    D = data.get('D')[:, None]
    text_length = data.get('text_length')

    return X, D, text_length


def get_D_init(X, n_atoms, atom_support, strategy='patch', window=True,
               noise_level=0.1, random_state=None):
    """Compute an initial dictionary

    Parameters
    ----------
    X : ndarray, shape (n_channels, *signal_support)
        signal to be encoded.
    n_atoms: int and tuple
        Determine the shape of the dictionary.
    atom_support: tuple (int, int)
        support of the atoms
    strategy: str in {'patch', 'random'} (default: 'patch')
        Strategy to compute initial dictionary:
           - 'random': draw iid coefficients iid in [0, 1]
           - 'patch': draw patches from X uniformly without replacement.
    window: boolean (default: True)
        Whether or not the algorithm will use windowed dictionary.
    noise_level: float (default: .1)
        If larger than 0, add gaussian noise to the initial dictionary. This
        helps escaping sub-optimal state where one atom is used only in one
        place with strategy='patch'.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.

    Returns
    -------
    D_init : ndarray, shape (n_atoms, n_channels, *atom_support)
        initial dictionary
    """
    rng = check_random_state(random_state)

    n_channels = X.shape[0]
    if strategy == 'random':
        D_init = rng.rand(n_atoms, n_channels, *atom_support)
    elif strategy == 'patch':
        D_init = init_dictionary(X, n_atoms=n_atoms, atom_support=atom_support,
                                 random_state=rng)
    else:
        raise NotImplementedError('strategy should be one of {patch, random}')

    # normalize the atoms
    D_init = prox_d(D_init)

    # Add a small noise to extracted patches. does not have a large influence
    # on the random init.
    if noise_level > 0:
        noise_level_ = noise_level * D_init.std(axis=(-1, -2), keepdims=True)
        noise = noise_level_ * rng.randn(*D_init.shape)
        D_init = prox_d(D_init + noise)

    # If the algorithm is windowed, correctly initiate the dictionary
    if window:
        atom_support = D_init.shape[-2:]
        tw = tukey_window(atom_support)[None, None]
        D_init *= tw

    return D_init


def multi_channel_2d_correlate(dk, pat):
    return np.sum([signal.correlate(dk_c, pat_c, mode='full')
                   for dk_c, pat_c in zip(dk, pat)], axis=0)


def evaluate_D_hat(patterns, D_hat):
    patterns, D_hat = patterns.copy(), D_hat.copy()

    axis = (2, 3)
    patterns /= np.linalg.norm(patterns, ord='f', axis=axis, keepdims=True)
    D_hat /= np.linalg.norm(D_hat, ord='f', axis=axis, keepdims=True)

    corr = np.array([
        [multi_channel_2d_correlate(d_k, pat).max() for d_k in D_hat]
        for pat in patterns
    ])
    return corr


def compute_best_assignment(corr):
    i, j = linear_sum_assignment(corr, maximize=True)
    return corr[i, j].mean()


def evaluate_one(fname, std, n_atoms=None, reg=.2, n_jobs=10, window=True,
                 random_state=None):
    rng = check_random_state(random_state)

    i = fname.split('.')[0].split('_')[-1]

    X, D, text_length = get_input(fname)
    X += std * X.std() * rng.randn(*X.shape)

    if 'PAMI' in fname:
        D = np.pad(D, [(0, 0), (0, 0), (4, 4), (4, 4)])
    n_atoms = D.shape[0] if n_atoms is None else n_atoms
    atom_support = np.array(D.shape[-2:])

    tag = f"l={text_length}_std={std}_{i}"
    if window:
        tag = f"{tag}_win"

    D_init = get_D_init(X, n_atoms, atom_support, strategy='patch',
                        window=window, noise_level=.1,
                        random_state=rng)

    D_rand = prox_d(rng.rand(*D_init.shape))
    corr_rand = evaluate_D_hat(D, D_rand)
    score_rand = corr_rand.max(axis=1).mean()
    score_rand_2 = compute_best_assignment(corr_rand)
    print(f"[{tag}] Rand score: {score_rand}, {score_rand_2}")

    corr_init = evaluate_D_hat(D, D_init)
    score_init = corr_init.max(axis=1).mean()
    score_init_2 = compute_best_assignment(corr_init)
    print(f"[{tag}] Init score: {score_init}, {score_init_2}")

    D_cdl, meta_cdl = compute_cdl(X, n_atoms, atom_support, D_init, reg=.2,
                                  window=window, n_jobs=n_jobs)
    corr_cdl = evaluate_D_hat(D, D_cdl)
    score_cdl = corr_cdl.max(axis=1).mean()
    score_cdl_2 = compute_best_assignment(corr_cdl)
    print(f"[{tag}] CDL score: {score_cdl}, {score_cdl_2}")

    D_dl, meta_dl = compute_dl(X, n_atoms, atom_support, reg=1e-1,
                               n_jobs=n_jobs)
    corr_dl = evaluate_D_hat(D, D_dl)
    score_dl = corr_dl.max(axis=1).mean()
    score_dl_2 = compute_best_assignment(corr_dl)
    print(f"[{tag}] DL score: {score_dl}, {score_dl_2}")

    return dict(
        text_length=int(text_length), noise_level=std, D=D,
        D_rand=D_rand, corr_rand=corr_rand, score_rand=score_rand,
        D_init=D_init, corr_init=corr_init, score_init=score_init,
        D_cdl=D_cdl, corr_cdl=corr_cdl, score_cdl=score_cdl,
        D_dl=D_dl, corr_dl=corr_dl, score_dl=score_dl,
        score_rand_2=score_rand_2, score_init_2=score_init_2,
        score_cdl_2=score_cdl_2, score_dl_2=score_dl_2,
        meta_dl=meta_dl, meta_cdl=meta_cdl, n_atoms=n_atoms,
        filename=fname,
    )


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--n-jobs', '-n', type=int, default=40,
                        help='Number of workers')
    parser.add_argument('--n-atoms', '-K', type=int, default=None,
                        help='Number of atoms to learn')
    parser.add_argument('--window', action='store_true',
                        help='If this flag is set, apply a window on the atoms'
                        ' to promote border to 0.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the random number generator. '
                        'Default to None.')
    parser.add_argument('--PAMI', action='store_true',
                        help='Run the CDL on text with PAMI letters.')
    args = parser.parse_args()

    rng = check_random_state(args.seed)

    if args.PAMI:
        from benchmarks.dicodile_text_plot import plot_dictionary
        std = 3
        std = .0001
        fname = 'text_4_5000_PAMI.npz'
        res_item = evaluate_one(
            fname, std, n_atoms=args.n_atoms,
            n_jobs=args.n_jobs, window=args.window, random_state=rng
        )
        now = datetime.now()
        t_tag = now.strftime('%y-%m-%d_%Hh%M')
        save_name = OUTPUT_DIR / f'{BASE_FILE_NAME}_PAMI_{t_tag}.pkl'
        results = pd.DataFrame([res_item])
        results.to_pickle(save_name, protocol=4)
        print(f'Saved results in {save_name}')

        plot_dictionary(res=res_item)

        raise SystemExit(0)

    INPUTS = [
        'text_5_150_0.npz', 'text_5_150_1.npz', 'text_5_150_2.npz',
        'text_5_150_3.npz', 'text_5_150_4.npz', 'text_5_150_5.npz',
        'text_5_150_6.npz', 'text_5_150_7.npz', 'text_5_150_8.npz',
        'text_5_150_9.npz',
        'text_5_360_0.npz', 'text_5_360_1.npz', 'text_5_360_2.npz',
        'text_5_360_3.npz', 'text_5_360_4.npz', 'text_5_360_5.npz',
        'text_5_360_6.npz', 'text_5_360_7.npz', 'text_5_360_8.npz',
        'text_5_360_9.npz',
        'text_5_866_0.npz', 'text_5_866_1.npz', 'text_5_866_2.npz',
        'text_5_866_3.npz', 'text_5_866_4.npz', 'text_5_866_5.npz',
        'text_5_866_6.npz', 'text_5_866_7.npz', 'text_5_866_8.npz',
        'text_5_866_9.npz',
        'text_5_2081_0.npz', 'text_5_2081_1.npz', 'text_5_2081_2.npz',
        'text_5_2081_3.npz', 'text_5_2081_4.npz', 'text_5_2081_5.npz',
        'text_5_2081_6.npz', 'text_5_2081_7.npz', 'text_5_2081_8.npz',
        'text_5_2081_9.npz',
        'text_5_5000_0.npz', 'text_5_5000_1.npz', 'text_5_5000_2.npz',
        'text_5_5000_3.npz', 'text_5_5000_4.npz', 'text_5_5000_5.npz',
        'text_5_5000_6.npz', 'text_5_5000_7.npz', 'text_5_5000_8.npz',
        'text_5_5000_9.npz'
    ]

    results = []
    # all_noise_levels = [3, 2, 1, .5, .1]
    all_noise_levels = [3, 2, .1]

    for i, fname in enumerate(reversed(INPUTS)):
        for std in all_noise_levels:
            res_item = evaluate_one(
                fname, std, n_atoms=args.n_atoms, n_jobs=args.n_jobs,
                window=args.window, random_state=rng
            )
            results.append(res_item)

    now = datetime.now()
    t_tag = now.strftime('%y-%m-%d_%Hh%M')
    save_name = OUTPUT_DIR / f'{BASE_FILE_NAME}_{t_tag}.pkl'

    results = pd.DataFrame(results)
    results.to_pickle(save_name, protocol=4)
    print(f'Saved results in {save_name}')
