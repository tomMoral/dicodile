import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from dicodile.utils import check_random_state

# Import for online DL
import spams
from sklearn.feature_extraction.image import extract_patches_2d

# Import to initiate the dictionary
from dicodile.utils.dictionary import prox_d
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.dictionary import init_dictionary


# Caching utility
from joblib import Memory

memory = Memory(location='.', verbose=0)


BASE_FILE_NAME = Path(__file__).with_suffix('').name
OUTPUT_DIR = Path('benchmarks_results')
DATA_DIR = Path('../..') / 'data' / 'images' / 'text'


@memory.cache
def compute_dl(X, n_atoms, atom_support, reg=.2,
               max_patches=2_000_000, n_jobs=10):
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
    batch_size = 512

    n_channels, *_ = X.shape
    assert n_channels == 1, (
        f'n_channels larger than 1 is not supported. Got {n_channels}'
    )
    X = X[0]

    # extract 2d patches from the image
    X_dl = extract_patches_2d(X, atom_support, max_patches=max_patches)
    n_patches = X_dl.shape[0]

    X_dl = X_dl.reshape(n_patches, -1)
    norm = np.linalg.norm(X_dl, axis=1)
    mask = norm != 0
    X_dl = X_dl[mask]
    X_dl /= norm[mask][:, None]

    # artificially increase the size of the epochs as spams segfaults for
    # the real number of patches
    if n_patches == max_patches:
        print("MAX_PATCHES!!!!", max_patches)
        n_patches = 4_736_160

    n_iter = 10_000
    n_epoch = n_iter * batch_size / n_patches
    meta = dict(lambda1=reg, iter=n_iter, mode=2, posAlpha=True, posD=False)

    # Learn the dictionary with spams
    t_start = time.time()
    spams.trainDL(np.asfortranarray(X_dl.T, dtype=np.float),
                  numThreads=n_jobs, batchsize=batch_size,
                  K=n_atoms, **meta, verbose=False).T
    runtime = time.time() - t_start

    return runtime, n_epoch


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


def evaluate_one(fname, std, n_atoms=None, reg=.2, n_jobs=10, window=True,
                 random_state=None):
    rng = check_random_state(random_state)

    X, D, text_length = get_input(fname)
    X += std * X.std() * rng.randn(*X.shape)

    n_atoms = D.shape[0] if n_atoms is None else n_atoms
    atom_support = np.array(D.shape[-2:])

    runtime, n_iter = compute_dl(
        X, n_atoms, atom_support, reg=.2, n_jobs=n_jobs
    )
    if runtime is None:
        print(f'[ODL-{n_jobs}] failed')
    else:
        print(f'[ODL-{n_jobs}] runtime/iter : {runtime / n_iter:.2f}s')

    return dict(
        text_length=int(text_length), noise_level=std,
        X_shape=X.shape, D_shape=D.shape, filename=fname,
        runtime=runtime, n_jobs=n_jobs, n_iter=n_iter
    )


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
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
    noise_level = 3
    random_states = [rng.randint(int(1e6)) for _ in range(len(INPUTS))]

    for n_jobs in [16, 4, 1]:
        for i, fname in enumerate(reversed(INPUTS)):
            print("Computing:", i)
            res_item = evaluate_one(
                fname, noise_level, n_atoms=args.n_atoms, n_jobs=n_jobs,
                window=args.window, random_state=random_states[i]
            )
            results.append(res_item)

    now = datetime.now()
    t_tag = now.strftime('%y-%m-%d_%Hh%M')
    save_name = OUTPUT_DIR / f'{BASE_FILE_NAME}_{t_tag}.pkl'

    results = pd.DataFrame(results)
    results.to_pickle(save_name, protocol=4)
    print(f'Saved results in {save_name}')
