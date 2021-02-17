"""Compare scaling of DICOD and DiCoDiLe_Z on a grid vs scaling in 2D.

Author: tommoral <thomas.moreau@inria.fr>
"""
import os
import pandas
import itertools
import numpy as np
from pathlib import Path
from joblib import Memory
import matplotlib.pyplot as plt
from collections import namedtuple

from benchmarks.parallel_resource_balance import delayed
from benchmarks.parallel_resource_balance import ParallelResourceBalance

from dicodile.update_z.dicod import dicod
from dicodile.data.images import fetch_mandrill
from dicodile.utils import check_random_state
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.dictionary import init_dictionary


###########################################
# Helper functions and constants
###########################################

# Maximal number to generate seeds
MAX_INT = 4294967295


# File names constants to save the results
SAVE_DIR = Path("benchmarks_results")
BASE_FILE_NAME = os.path.basename(__file__)
SAVE_FILE_BASENAME = SAVE_DIR / BASE_FILE_NAME.replace('.py', '{}')


def get_save_file_name(ext='pkl', **kwargs):
    file_name = str(SAVE_FILE_BASENAME).format("{suffix}.{ext}")
    suffix = ""
    for k, v in kwargs.items():
        suffix += f"_{k}={str(v).replace('.', '-')}"

    return file_name.format(suffix=suffix, ext=ext)


# Constants for logging in console.
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


# Add color output to consol logging.
def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


###############################################
# Helper function to cache computations
# and make the benchmark robust to failures
###############################################

# Caching utility from joblib
mem = Memory(location='.', verbose=0)

# Result item, to help with the pandas.DataFrame construction
ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_workers', 'strategy', 'tol',
    'dicod_args', 'random_state', 'sparsity', 'iterations', 'runtime',
    't_init', 't_run', 'n_updates', 't_select', 't_update'])


@mem.cache(ignore=['dicod_args'])
def run_one_scaling_2d(n_atoms, atom_support, reg, n_workers, strategy, tol,
                       dicod_args, random_state):
    tag = f"[{strategy} - {reg:.0e} - {random_state[0]}]"
    random_state = random_state[1]

    # Generate a problem
    print(colorify(79*"=" + f"\n{tag} Start with {n_workers} workers\n" +
                   79*"="))
    X = fetch_mandrill()
    D = init_dictionary(X, n_atoms, atom_support, random_state=random_state)
    reg_ = reg * get_lambda_max(X, D).max()

    z_hat, *_, run_statistics = dicod(
        X, D, reg=reg_, strategy=strategy, n_workers=n_workers, tol=tol,
        **dicod_args)

    runtime = run_statistics['runtime']
    sparsity = len(z_hat.nonzero()[0]) / z_hat.size
    print(colorify('=' * 79 + f"\n{tag} End with {n_workers} workers for reg="
                   f"{reg:.0e} in {runtime:.1e}\n" + "=" * 79, color=GREEN))

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      n_workers=n_workers, strategy=strategy, tol=tol,
                      dicod_args=dicod_args, random_state=random_state,
                      sparsity=sparsity, **run_statistics)


#######################################
# Function to run the benchmark
#######################################

def run_scaling_benchmark(max_n_workers, n_rep=1, random_state=None):
    '''Run DICOD with different n_workers for a 2D problem.
    '''

    # Parameters to generate the simulated problems
    n_atoms = 5
    atom_support = (8, 8)
    rng = check_random_state(random_state)

    # Parameters for the algorithm
    tol = 1e-3
    dicod_args = dict(z_positive=False, soft_lock='border', timeout=None,
                      max_iter=int(1e9), verbose=1)

    # Generate the list of parameter to call
    reg_list = [5e-1, 2e-1, 1e-1]
    list_n_workers = np.unique(np.logspace(0, np.log10(256), 15, dtype=int))
    list_n_workers = [n if n != 172 else 169 for n in list_n_workers]
    list_n_workers += [18*18, 20*20]
    list_strategies = ['lgcd', 'gcd']
    list_random_states = list(enumerate(rng.randint(MAX_INT, size=n_rep)))

    assert np.max(list_n_workers) < max_n_workers, (
        f"This benchmark need to have more than {list_n_workers.max()} to run."
        f" max_n_workers was set to {max_n_workers}, which is too low."
    )

    it_args = itertools.product(list_n_workers, reg_list, list_strategies,
                                list_random_states)

    # run the benchmark
    run_one = delayed(run_one_scaling_2d)
    results = ParallelResourceBalance(max_workers=max_n_workers)(
        run_one(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                n_workers=n_workers, strategy=strategy, tol=tol,
                dicod_args=dicod_args, random_state=random_state)
        for (n_workers, reg, strategy, random_state) in it_args)

    # Save the results as a DataFrame
    results = pandas.DataFrame(results)
    results.to_pickle(get_save_file_name(ext='pkl'))


###############################################
# Function to plot the benchmark result
###############################################

def plot_scaling_benchmark():
    df = pandas.read_pickle(get_save_file_name(ext='pkl'))
    import matplotlib.lines as lines
    handles_lmbd = {}
    handles_strategy = {}
    fig = plt.figure(figsize=(6, 3))
    fig.patch.set_alpha(0)

    ax = plt.subplot()

    colors = ['C0', 'C1', 'C2']
    regs = df['reg'].unique()
    regs.sort()
    for reg, c in zip(regs, colors):
        for strategy, style in [('LGCD', '-'), ('GCD', '--')]:
            s = strategy.lower()
            this_df = df[(df.reg == reg) & (df.strategy == s)]
            curve = this_df.groupby('n_workers').runtime
            runtimes = curve.mean()
            runtime_std = curve.std()

            print(runtimes.index.max())
            plt.fill_between(runtimes.index, runtimes - runtime_std,
                             runtimes + runtime_std, alpha=.1)
            plt.loglog(runtimes.index, runtimes, label=f"{strategy}_{reg:.2f}",
                       linestyle=style, c=c)
            color_handle = lines.Line2D(
                [], [], linestyle='-', c=c, label=f"${reg:.1f}\\lambda_\\max$")
            style_handle = lines.Line2D(
                [], [], linestyle=style, c='k', label=f"{strategy}")
            handles_lmbd[reg] = color_handle
            handles_strategy[strategy] = style_handle

            # min_workers = this_df.n_workers.min()
            # max_workers = this_df.n_workers.max()
            # t = np.logspace(np.log10(min_workers), np.log10(max_workers),
            #                 6)
            # p = 1
            # R0 = runtimes.loc[min_workers]
            # scaling = lambda t: R0 / (t / min_workers) ** p  # noqa: E731
            # plt.plot(t, scaling(t), 'k--')
    plt.xlim((1, runtimes.index.max()))
    plt.ylim((2e1, 2e4))
    # plt.xticks(n_workers, n_workers, fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.minorticks_off(axis='y')
    plt.xlabel("# workers $W$", fontsize=12)
    plt.ylabel("Runtime [sec]", fontsize=12)
    plt.grid(True, which="both", axis='x')
    plt.grid(True, which="major", axis='y')

    # keys = list(handles.keys())
    # keys.sort()
    # handles = [handles[k] for k in keys]
    legend_lmbd = plt.legend(handles=handles_lmbd.values(), loc=1,
                             fontsize=14)
    plt.legend(handles=handles_strategy.values(), loc=3, ncol=2, fontsize=14)
    ax.add_artist(legend_lmbd)
    plt.tight_layout()
    plt.savefig(get_save_file_name(ext='pdf'), dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Benchmark scaling performance for DICOD')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the result of the benchmark')
    parser.add_argument('--n-rep', type=int, default=10,
                        help='Number of repetition to average to compute the '
                        'average running time.')
    parser.add_argument('--max-workers', type=int, default=75,
                        help='Maximal number of workers used.')
    args = parser.parse_args()

    random_state = 2727

    if args.plot:
        plot_scaling_benchmark()
    else:
        run_scaling_benchmark(max_n_workers=args.max_workers, n_rep=args.n_rep,
                              random_state=random_state)
