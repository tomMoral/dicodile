import os
import time
import pandas
import itertools
import numpy as np
from pathlib import Path
from joblib import Memory
import matplotlib.pyplot as plt
from collections import namedtuple


from dicodile.update_z.dicod import dicod
from dicodile.utils import check_random_state
from dicodile.data.simulate import simulate_data
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.viz import RotationAwareAnnotation


MAX_INT = 4294967295
COLOR = ['C2', 'C1', 'C0']
SAVE_DIR = Path("benchmarks_results")
BASE_FILE_NAME = os.path.basename(__file__)
SAVE_FILE_NAME = str(SAVE_DIR / BASE_FILE_NAME.replace('.py', '{}'))

mem = Memory(location='.', verbose=0)


###################################
# Helper function for outputs
###################################

# Constants for logging in console.
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


# Add color output to consol logging.
def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


# Result item to create the DataFrame in a consistent way.
ResultItem = namedtuple('ResultItem', [
    'n_workers', 'strategy', 'reg', 'n_times', 'tol', 'soft_lock',
    'meta', 'random_state', 'iterations', 'runtime', 't_init', 't_run',
    'n_updates', 't_select', 't_update'])


###############################################
# Helper function to cache computations
# and make the benchmark robust to failures
###############################################

@mem.cache(ignore=['dicod_args'])
def run_one(n_workers, strategy, reg, n_times, tol, soft_lock, dicod_args,
            n_times_atom, n_atoms, n_channels, noise_level, random_state):

    tag = f"[{strategy} - {n_times} - {reg:.0e} - {random_state}]"

    t_start_generation = time.time()
    print(colorify(f"{tag} Signal generation..."), end='', flush=True)
    X, D_hat = simulate_data(n_times=n_times, n_times_atom=n_times_atom,
                             n_atoms=n_atoms, n_channels=n_channels,
                             noise_level=noise_level,
                             random_state=random_state)
    lmbd_max = get_lambda_max(X, D_hat)
    reg_ = reg * lmbd_max
    print(colorify(f"done ({time.time() - t_start_generation:.3f}s)."))

    n_seg = 1
    strategy_ = strategy
    if strategy == 'lgcd':
        n_seg = 'auto'
        strategy_ = "greedy"

    *_, run_statistics = dicod(X, D_hat, reg_, n_workers=n_workers, tol=tol,
                               strategy=strategy_, n_seg=n_seg,
                               soft_lock=soft_lock, **dicod_args)
    meta = dicod_args.copy()
    meta.update(n_atoms=n_atoms, n_times_atom=n_times_atom,
                n_channels=n_channels, noise_level=noise_level)
    runtime = run_statistics['runtime']

    print(colorify('=' * 79 +
                   f"\n{tag} End with {n_workers} workers in {runtime:.1e}\n" +
                   "=" * 79, color=GREEN))

    return ResultItem(n_workers=n_workers, strategy=strategy, reg=reg,
                      n_times=n_times, tol=tol, soft_lock=soft_lock, meta=meta,
                      random_state=random_state, **run_statistics)


###############################################
# Benchmarking function
###############################################

def run_scaling_1d_benchmark(strategies, n_rep=1, max_workers=75, timeout=None,
                             soft_lock='none', list_n_times=[151, 750],
                             list_reg=[2e-1, 5e-1], random_state=None):
    '''Run DICOD strategy for a certain problem with different value
    for n_workers and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    strategies: list of str in { 'greedy', 'lgcd', 'random' }
        Algorithm to run the benchmark for
    n_rep: int (default: 10)
        Number of repetition to average the results.
    max_workers: int (default: 75)
        The strategy will be run on problems with a number
        of cores varying from 1 to max_workers in a log2 scale
    soft_lock: str in {'none', 'border'}
        Soft-lock mechanism to use in dicod
    timeout: int (default: None)
        maximal running time for DICOD. The default timeout
        is 2 hours
    list_n_times: list of int
        Size of the generated problems
    list_reg: list of float
        Regularization parameter of the considered problem
    list_tol: list of float
        Tolerance parameter used in DICOD.
    random_state: None or int or RandomState
        Seed for the random number generator.
    '''
    rng = check_random_state(random_state)

    # Parameters to generate the simulated problems
    n_times_atom = 250
    n_atoms = 25
    n_channels = 7
    noise_level = 1

    # Parameters for the algorithm
    tol = 1e-8
    dicod_args = dict(timing=False, timeout=7200,
                      max_iter=int(5e8), verbose=2)

    # Get the list of parameter to call
    list_n_workers = np.unique(np.logspace(0, np.log10(max_workers), 15,
                               dtype=int))
    list_n_workers = list_n_workers[::-1]
    list_seeds = [rng.randint(MAX_INT) for _ in range(n_rep)]
    strategies = [s[0] for s in strategies]
    list_args = itertools.product(list_n_workers, strategies, list_reg,
                                  list_n_times, list_seeds)

    # Run the computation
    results = [run_one(n_workers=n_workers, strategy=strategy, reg=reg,
                       n_times=n_times, tol=tol, soft_lock=soft_lock,
                       dicod_args=dicod_args, n_times_atom=n_times_atom,
                       n_atoms=n_atoms, n_channels=n_channels,
                       noise_level=noise_level, random_state=random_state)
               for (n_workers, strategy, reg,
                    n_times, random_state) in list_args]

    # Save the results as a DataFrame
    results = pandas.DataFrame(results)
    results.to_pickle(SAVE_FILE_NAME.format('.pkl'))


###############################################
# Function to plot the benchmark result
###############################################

def plot_scaling_1d_benchmark():
    config = {
        'greedy': {
            'style': 'C1-o',
            'label': "DICOD",
            'scaling': 2
        },
        'lgcd': {
            'style': 'C0-s',
            'label': 'Dicodile$_Z$'
        }
    }

    full_df = pandas.read_pickle(SAVE_FILE_NAME.format('.pkl'))
    for T in full_df.n_times.unique():
        T_df = full_df[full_df.n_times == T]
        for reg in T_df.reg.unique():
            plt.figure(figsize=(6, 3))
            ylim = 1e100, 0
            reg_df = T_df[T_df.reg == reg]
            for strategy in reg_df.strategy.unique():
                df = reg_df[reg_df.strategy == strategy]
                curve = df.groupby('n_workers').mean()
                ylim = (min(ylim[0], curve.runtime.min()),
                        max(ylim[1], curve.runtime.max()))

                label = config[strategy]['label']
                style = config[strategy]['style']
                plt.loglog(curve.index, curve.runtime, style, label=label,
                           markersize=8)

                # Plot scaling
                t = np.linspace(df.n_workers.min(), df.n_workers.max())
                p = config[strategy].get('scaling', 1)
                R0 = np.max(curve.runtime)
                plt.plot(t, R0 / t ** p, 'k--')

                tt = 2
                eps = 1e-10
                text = "linear" if p == 1 else "quadratic"
                anchor_pt = np.array([tt, R0 / tt**p])
                next_pt = np.array([tt + eps, R0 / (tt + eps)**p])

                RotationAwareAnnotation(
                    text, anchor_pt=anchor_pt, next_pt=next_pt,
                    xytext=(0, -15), textcoords="offset points", fontsize=12,
                    horizontalalignment='center', verticalalignment='center')

            # Add a line on scale improvement limit
            # plt.vlines(T / 4, 1e-10, 1e10, 'g', '-.')

            # Set the axis limits
            plt.xlim(t.min(), t.max())
            ylim = (10 ** int(np.log10(ylim[0]) - 1),
                    min(10 ** int(np.log10(ylim[1]) + 1), 3 * ylim[1]))
            plt.ylim(ylim)

            # Add grids to improve readability
            plt.grid(True, which='both', axis='x', alpha=.5)
            plt.grid(True, which='major', axis='y', alpha=.5)

            # Add axis labels
            plt.ylabel("Runtime [sec]")
            plt.xlabel("# workers $W$")
            plt.legend()

            # Save the figures
            suffix = f"_T={T}_reg={str(reg).replace('.', '-')}.pdf"
            plt.savefig(SAVE_FILE_NAME.format(suffix), dpi=300,
                        bbox_inches='tight', pad_inches=0)
    plt.close('all')


###########################
#    Main script
###########################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Benchmarking script to test the scaling '
                                     'of Dicodile_Z with the number of cores '
                                     'for 1D convolutional sparse coding')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the results of the benchmark')
    parser.add_argument('--n-rep', type=int, default=10,
                        help='Number of repetition to average to compute the '
                        'average running time.')
    parser.add_argument('--max-workers', type=int, default=75,
                        help='Maximal number of workers used.')
    args = parser.parse_args()

    random_state = 422742

    soft_lock = 'none'
    strategies = [
        ('greedy', 'Greedy', 's-'),
        # ('cyclic', 'Cyclic', "h-"),
        ('lgcd', "LGCD", 'o-')
    ]
    list_reg = [2e-1, 5e-1]
    list_n_times = [201, 500, 1000]

    if args.plot:
        plot_scaling_1d_benchmark()
    else:
        run_scaling_1d_benchmark(
            strategies, n_rep=args.n_rep, max_workers=args.max_workers,
            soft_lock=soft_lock, list_n_times=list_n_times, list_reg=list_reg,
            random_state=random_state)
