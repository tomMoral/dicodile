"""Compare scaling of DICOD and DiCoDiLe_Z on a grid vs scaling in 1D.

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
from dicodile.data.images import get_mandril
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


# Result item to create the DataFrame in a consistent way.
ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_workers', 'grid', 'tol', 'soft_lock',
    'random_state', 'dicod_args', 'sparsity', 'iterations', 'runtime',
    't_init', 't_run', 'n_updates', 't_select', 't_update'])


@mem.cache(ignore=['dicod_args'])
def run_one_grid(n_atoms, atom_support, reg, n_workers, grid, tol,
                 soft_lock, dicod_args, random_state):

    tag = f"[{soft_lock} - {reg:.0e} - {random_state[0]}]"
    random_state = random_state[1]

    # Generate a problem
    print(colorify(79*"=" + f"\n{tag} Start with {n_workers} workers\n" +
                   79*"="))
    X = get_mandril()
    D = init_dictionary(X, n_atoms, atom_support, random_state=random_state)
    reg_ = reg * get_lambda_max(X, D).max()

    if grid:
        w_world = 'auto'
    else:
        w_world = n_workers

    z_hat, *_, run_statistics = dicod(
        X, D, reg=reg_, n_seg='auto', strategy='greedy', w_world=w_world,
        n_workers=n_workers, timing=False, tol=tol,
        soft_lock=soft_lock, **dicod_args)

    runtime = run_statistics['runtime']
    sparsity = len(z_hat.nonzero()[0]) / z_hat.size

    print(colorify("=" * 79 + f"\n{tag} End for {n_workers} workers "
                   f"in {runtime:.1e}\n" + "=" * 79, color=GREEN))

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      n_workers=n_workers, grid=grid, tol=tol,
                      soft_lock=soft_lock, random_state=random_state,
                      dicod_args=dicod_args, sparsity=sparsity,
                      **run_statistics)


#######################################
# Function to run the benchmark
#######################################

def run_scaling_grid(n_rep=1, max_workers=225, random_state=None):
    '''Run DICOD with different n_workers on a grid and on a line.
    '''
    # Parameters to generate the simulated problems
    n_atoms = 5
    atom_support = (8, 8)
    rng = check_random_state(random_state)

    # Parameters for the algorithm
    tol = 1e-4
    dicod_args = dict(z_positive=False, timeout=None, max_iter=int(1e9),
                      verbose=1)

    # Generate the list of parameter to call
    reg_list = [5e-1, 2e-1, 1e-1]
    list_soft_lock = ['border']  # , 'corner']
    list_n_workers = np.unique(np.logspace(0, np.log10(15), 20, dtype=int))**2
    list_random_states = enumerate(rng.randint(MAX_INT, size=n_rep))

    it_args = itertools.product(reg_list, [True, False], list_n_workers,
                                list_soft_lock, list_random_states)

    # Filter out the arguments where the algorithm cannot run because there
    # is too many workers.
    it_args = [args for args in it_args if args[1] or args[2] <= 36]
    it_args = [args if args[1] or args[2] < 32 else (*args[:2], 32, *args[3:])
               for args in it_args]

    # run the benchmark
    run_one = delayed(run_one_grid)
    results = ParallelResourceBalance(max_workers=max_workers)(
        run_one(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                n_workers=n_workers, grid=grid, tol=tol, soft_lock=soft_lock,
                dicod_args=dicod_args, random_state=random_state)
        for (reg, grid, n_workers, soft_lock, random_state) in it_args)

    # Save the results as a DataFrame
    results = pandas.DataFrame(results)
    results.to_pickle(get_save_file_name(ext='pkl'))


###############################################
# Function to plot the benchmark result
###############################################

def plot_scaling_benchmark():
    full_df = pandas.read_pickle(get_save_file_name(ext='pkl'))

    list_reg = list(np.unique(full_df.reg)) + ['all']
    for reg in list_reg:
        fig = plt.figure(figsize=(6, 3))
        fig.patch.set_alpha(0)
        for name, use_grid in [("Linear Split", False), ("Grid Split", True)]:
            curve = []
            if reg == 'all':
                df = full_df[full_df.grid == use_grid]
            else:
                df = full_df[(full_df.grid == use_grid) & (full_df.reg == reg)]
            curve = df.groupby('n_workers').runtime.mean()
            plt.loglog(curve.index, curve, label=name)

        ylim = plt.ylim()
        plt.vlines(512 / (8 * 2), *ylim, colors='g', linestyles='-.')
        plt.ylim(ylim)
        plt.legend(fontsize=14)
        # plt.xticks(n_workers, n_workers)
        plt.grid(which='both')
        plt.xlim((1, 225))
        plt.ylabel("Runtime [sec]", fontsize=12)
        plt.xlabel("# workers $W$", fontsize=12)
        plt.tight_layout()

        fig.savefig(get_save_file_name(ext='pdf', reg=reg), dpi=300,
                    bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Benchmark scaling performance for DICOD')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the result of the benchmark.')
    parser.add_argument('--max-workers', type=int, default=225,
                        help='Maximal number of workers possible to use.')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='Number of repetition to average the runtime.')
    args = parser.parse_args()

    random_state = 4242

    if args.plot:
        plot_scaling_benchmark()
    else:
        run_scaling_grid(n_rep=args.n_rep, max_workers=args.max_workers,
                         random_state=random_state)
