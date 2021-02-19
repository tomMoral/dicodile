import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt

from dicodile.utils import check_random_state

# Import for CDL
from dicodile.dicodile import dicodile

# Import to initiate the dictionary
from dicodile.utils.dictionary import prox_d
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.dictionary import init_dictionary

# Plotting utils
from benchmarks.benchmark_utils import get_last_file
from benchmarks.benchmark_utils import mk_legend_handles

# Caching utility
from joblib import Memory

memory = Memory(location='.', verbose=0)


# Matplotlib config
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


BASE_FILE_NAME = Path(__file__).with_suffix('').name
OUTPUT_DIR = Path('benchmarks_results')
DATA_DIR = Path('../..') / 'data' / 'images' / 'text'


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

    meta = dict(
        reg=reg, tol=1e-3, z_positive=True, n_iter=100, window=window,
        strategy='greedy', dicod_kwargs={'timeout': 3600},
    )

    # fit the dictionary with dicodile
    t_start = time.time()
    pobj, times, D_hat, z_hat = dicodile(
        X_0, D_init, n_workers=n_jobs, w_world='auto',
        **meta, raise_on_increase=True, verbose=1,
    )
    runtime_real = time.time() - t_start
    runtime = np.sum(times)
    return runtime, runtime_real, len(pobj)


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

    i = fname.split('.')[0].split('_')[-1]

    X, D, text_length = get_input(fname)
    X += std * X.std() * rng.randn(*X.shape)

    n_atoms = D.shape[0] if n_atoms is None else n_atoms
    atom_support = np.array(D.shape[-2:])

    tag = f"l={text_length}_std={std}_{i}"
    if window:
        tag = f"{tag}_win"

    D_init = get_D_init(X, n_atoms, atom_support, strategy='patch',
                        window=window, noise_level=.1,
                        random_state=rng)

    runtime, runtime_real, n_iter = compute_cdl(
        X, n_atoms, atom_support, D_init, reg=.2, window=window, n_jobs=n_jobs
    )
    print(f'[Dicodile-{n_jobs}] runtime: {runtime:.2f}s ({runtime_real:.2f}s)')

    return dict(
        text_length=int(text_length), noise_level=std,
        X_shape=X.shape, D_shape=D.shape, filename=fname,
        runtime=runtime, runtime_real=runtime_real, n_jobs=n_jobs,
        n_iter=n_iter
    )


def get_label(name):
    return ' / '.join([w.capitalize() for w in name.split('_')])


def plot_results():

    # Load the results
    fname = get_last_file(OUTPUT_DIR, f'{BASE_FILE_NAME}_*.pkl')
    df = pd.read_pickle(fname)
    df['n_pixels'] = df['X_shape'].apply(np.prod)
    df['runtime_iter'] = df['runtime'] / ((df['n_iter'] - 1) // 2)

    fname_odl = get_last_file(OUTPUT_DIR, 'odl_text_runtime_*.pkl')
    df_odl = pd.read_pickle(fname_odl)
    df_odl['n_pixels'] = df_odl['X_shape'].apply(np.prod)
    df_odl['runtime_iter'] = df_odl['runtime'] / df_odl['n_iter']

    # Define line style
    common_style = dict(markersize=10, lw=4)

    for c_to_plot in ['runtime', 'runtime_iter']:
        fig, ax = plt.subplots(figsize=(6.4, 3))
        fig.subplots_adjust(right=.98)
        for n_jobs, c in [(1, 'C0'), (4, 'C2'), (16, 'C1')]:
            for this_df, s, m in [(df, '-', 's'), (df_odl, '--', 'o')]:
                this_df = this_df.query('n_jobs == @n_jobs')
                curve = this_df.groupby('text_length')[
                    ['n_pixels', c_to_plot]
                ].median()
                err = this_df.groupby('text_length')[
                    [c_to_plot]
                ].quantile([0.1, 0.9])
                err = err.reorder_levels([1, 0]).sort_index()[c_to_plot]
                ax.fill_between(
                    curve['n_pixels'], err[0.1], err[0.9], facecolor=c,
                    alpha=.2
                )
                ax.loglog(
                    curve.set_index('n_pixels')[c_to_plot], color=c,
                    label=f'{n_jobs} workers', marker=m, linestyle=s,
                    **common_style
                )
        handles, labels = mk_legend_handles([
            dict(linestyle='-', marker='s', label='DiCoDiLe'),
            dict(color='C0', label='1 worker'),
            dict(linestyle='--', marker='o', label='ODL'),
            dict(color='C2', label='4 workers'),
            dict(alpha=0, label=None), dict(color='C1', label='16 workers'),
        ], **common_style, color='k')
        ax.legend(
            handles, labels,
            ncol=3, loc='center', bbox_to_anchor=(0, 1.12, 1, .05),
            fontsize=14
        )

        x_ticks = np.array([0.2, 0.5, 1, 2, 4.8]) * 1e6
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x/1e6:.1f}' for x in x_ticks])
        ax.set_xlabel('Image Size [Mpx]')
        ax.set_ylabel(f'{get_label(c_to_plot)} [sec]')
        ax.set_xlim(curve['n_pixels'].min(), curve['n_pixels'].max())
        plt.savefig(OUTPUT_DIR / f'dicodile_text_{c_to_plot}.pdf', dpi=300)

    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--max-workers', '-m', type=int, default=40,
                        help='Number of workers')
    parser.add_argument('--n-atoms', '-K', type=int, default=None,
                        help='Number of atoms to learn')
    parser.add_argument('--window', action='store_true',
                        help='If this flag is set, apply a window on the atoms'
                        ' to promote border to 0.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the random number generator. '
                        'Default to None.')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results.')
    args = parser.parse_args()

    if args.plot:
        plot_results()
        raise SystemExit(0)

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

    # results = ParallelResourceBalance(max_workers=args.max_workers)(
    #     delayed(evaluate_one)(
    #         fname, noise_level, n_atoms=args.n_atoms, n_jobs=n_jobs,
    #         window=args.window, random_state=rng
    #     )
    #     for n_jobs in reversed([1, 16, 32])
    #     for i, fname in enumerate(reversed(INPUTS))
    # )

    # for n_jobs in reversed([4, 1, 16]):
    #     for i, fname in enumerate(reversed(INPUTS)):
    for n_jobs in reversed([1]):
        for i, fname in reversed(list(enumerate(reversed(INPUTS)))):
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
