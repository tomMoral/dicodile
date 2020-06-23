"""Compare the coordinate selection strategies for Coordinate descent on CSC.

Author: tommoral <thomas.moreau@inria.fr>
"""
import os
import time
import pandas
import itertools
import numpy as np
from pathlib import Path
from joblib import Memory
import matplotlib.pyplot as plt
from collections import namedtuple
from joblib import delayed, Parallel

from threadpoolctl import threadpool_limits


from dicodile.update_z.dicod import dicod
from dicodile.utils import check_random_state
from dicodile.data.simulate import simulate_data

from dicodile.utils.plot_config import get_style


MAX_INT = 4294967295
COLOR = ['C2', 'C1', 'C0', 'C3', 'C4']
SAVE_DIR = Path("benchmarks_results")
BASE_FILE_NAME = os.path.basename(__file__)
SAVE_FILE_NAME = str(SAVE_DIR / BASE_FILE_NAME.replace('.py', '{}'))

# Constants for logging in console.
START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


# jobib cache to avoid loosing computations
mem = Memory(location='.', verbose=0)


###################################
# Helper function for outputs
###################################

def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


ResultItem = namedtuple('ResultItem', [
    'reg', 'strategy', 'tol', 'n_times', 'meta', 'random_state', 'pobj',
    'iterations', 'runtime', 't_init', 't_run', 'n_updates',
    't_select', 't_update'])


@mem.cache(ignore=['dicod_args'])
def run_one(n_times, n_times_atom, n_atoms, n_channels, noise_level,
            random_state, reg, tol, strategy, dicod_args):

    threadpool_limits(1)

    tag = f"[{strategy} - {n_times} - {reg}]"
    current_time = time.time() - START
    msg = f"\r{tag} started at T={current_time:.0f} sec"
    print(colorify(msg, BLUE))

    X, D_hat, lmbd_max = simulate_data(
        n_times, n_times_atom, n_atoms, n_channels, noise_level,
        random_state=random_state)
    reg_ = reg * lmbd_max

    n_seg = 1
    if strategy == 'lgcd':
        n_seg = 'auto'

    *_, pobj, run_statistics = dicod(X, D_hat, reg_, n_workers=1, tol=tol,
                                     strategy=strategy, n_seg=n_seg,
                                     **dicod_args)
    meta = dicod_args.copy()
    meta.update(n_times_atom=n_times_atom, n_atoms=n_atoms,
                n_channels=n_channels, noise_level=noise_level)

    duration = time.time() - START - current_time
    msg = (f"\r{tag} done in {duration:.0f} sec "
           f"at T={time.time() - START:.0f} sec")
    print(colorify(msg, GREEN))

    return ResultItem(reg=reg, strategy=strategy, tol=tol, n_times=n_times,
                      meta=meta, random_state=random_state, pobj=pobj,
                      **run_statistics)


def compare_strategies(strategies, n_rep=10, n_workers=4, timeout=7200,
                       list_n_times=[150, 750], list_reg=[1e-1, 5e-1],
                       random_state=None):
    '''Run DICOD strategy for a certain problem with different value
    for n_workers and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    strategies: list of str in { 'greedy', 'lgcd', 'random', 'cyclic'}
        Algorithm to run the benchmark for
    n_rep: int (default: 10)
       Number of repetition for each strategy to average.
    n_workers: int (default: 4)
        Number of jobs to run strategies in parallel.
    timeout: int (default: 7200)
        maximal runtime for each strategy. The default timeout
        is 2 hours.
    list_n_times: list of int
        Size of the generated problems.
    list_reg: list of float
        Regularization parameter of the considered problem.
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
    dicod_args = dict(timing=False, timeout=timeout, max_iter=int(5e8),
                      verbose=2)

    # Get the list of parameter to call
    list_seeds = [rng.randint(MAX_INT) for _ in range(n_rep)]
    strategies = [s[0] for s in strategies]
    list_args = itertools.product(strategies, list_reg, list_n_times,
                                  list_seeds)

    # Run the computation
    results = Parallel(n_workers=n_workers)(
        delayed(run_one)(n_times, n_times_atom, n_atoms, n_channels,
                         noise_level, random_state, reg, tol, strategy,
                         dicod_args)
        for strategy, reg, n_times, random_state in list_args)

    # Save the results as a DataFrame
    results = pandas.DataFrame(results)
    results.to_pickle(SAVE_FILE_NAME.format('.pkl'))


def plot_comparison_strategies(strategies):
    full_df = pandas.read_pickle(SAVE_FILE_NAME.format('.pkl'))

    list_n_times = full_df.n_times.unique()
    list_regs = full_df.reg.unique()[1:]

    # compute the width of the bars
    n_group = len(list_n_times)
    n_bar = len(strategies)
    width = 1 / ((n_bar + 1) * n_group - 1)

    configs = [
        {'outer': ('reg', list_regs, r'\lambda', r'\lambda_{max}'),
         'inner': ('n_times', list_n_times, 'T', 'L')},
        {'outer': ('n_times', list_n_times, 'T', 'L'),
         'inner': ('reg', list_regs, r'\lambda', r'\lambda_{max}')}
    ]

    for config in configs:
        out_name, out_list, *_ = config['outer']
        in_name, in_list, label, unit = config['inner']
        for out in out_list:
            fig = plt.figure(f"comparison CD -- {out_name}={out}",
                             figsize=(6, 3.15))
            ax_bar = fig.subplots()
            xticks, labels = [], []
            ylim = (1e10, 0)
            for i, in_ in enumerate(in_list):
                handles = []
                xticks.append(((i + .5) * (n_bar + 1)) * width)
                labels.append(f"${label} = {in_}{unit}$")
                for j, (strategy, name) in enumerate(strategies):
                    df = full_df[full_df[out_name] == out]
                    df = df[df[in_name] == in_]
                    df = df[df.strategy == strategy]
                    position = (i * (n_bar + 1) + j + 1) * width

                    t_run = df.t_run.to_numpy()
                    handles.append(ax_bar.bar(
                        position, height=np.median(t_run), width=width,
                        **get_style(strategy, 'hatch')
                        # facecolor=COLOR[j], label=name,
                        # hatch='//' if strategy == 'lgcd' else '')
                    ))
                    ax_bar.plot(np.ones_like(t_run) * position, t_run, 'k_')
                    ylim = (min(ylim[0], t_run.min()),
                            max(ylim[1], t_run.max()))
            ax_bar.set_ylabel("Runtime [sec]", fontsize=12)
            ax_bar.set_yscale('log')
            ax_bar.set_xticks(xticks)
            ax_bar.set_xticklabels(labels, fontsize=14)
            # ax_bar.set_ylim(ylim[0] / 5, 5 * ylim[1])
            ax_bar.set_ylim(.1, 1e5)
            ax_bar.legend(bbox_to_anchor=(0, 1.05, 1., .3), loc="lower left",
                          handles=handles, ncol=3, borderaxespad=0.,
                          fontsize=14)
            fig.tight_layout()
            out = str(out).replace('.', ',')
            for ext in ['png']:
                fig.savefig(SAVE_FILE_NAME.format(f"_{out_name}={out}.{ext}"),
                            dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the results of the benchmarl')
    parser.add_argument('--qp', action="store_true",
                        help='Plot the results of the benchmarl')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='Number of repetition to average to compute the '
                        'average running time.')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of worker to run the script.')
    args = parser.parse_args()

    random_state = 422742

    strategies = [
        ('cyclic', 'Cyclic'),
        # ('cyclic-r', 'Shuffle', "h-"),
        ('lgcd', "LGCD"),
        ('greedy', 'Greedy'),
        # ('random', 'Random', ">-"),
    ]

    if args.plot:
        plot_comparison_strategies(strategies)
    else:
        list_n_times = [200, 500, 1000]
        list_reg = [5e-2, 1e-1, 2e-1, 5e-1]
        compare_strategies(
            strategies, n_rep=args.n_rep, n_workers=args.n_workers,
            timeout=None, list_n_times=list_n_times, list_reg=list_reg,
            random_state=random_state)
