import time
import pandas
import itertools
import numpy as np
from joblib import Memory
import matplotlib.pyplot as plt
from collections import namedtuple


from dicodile.update_z.dicod import dicod
from dicodile.utils import check_random_state
from dicodile.data.simulate import simulate_data
from dicodile.utils.dictionary import get_lambda_max


MAX_INT = 4294967295
COLOR = ['C2', 'C1', 'C0']
SAVE_DIR = "benchmarks_results"

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
    results.to_pickle("benchmarks_results/scaling_1d.pkl")


def plot_scaling_1d_benchmark(strategies, list_n_times):

    # compute the width of the bars
    n_group = len(list_n_times)
    n_bar = len(strategies)
    width = 1 / ((n_bar + 1) * n_group - 1)

    fig = plt.figure('comparison CD', figsize=(6, 3.5))
    fig.patch.set_alpha(0)
    ax_bar = fig.subplots()
    xticks, labels = [], []
    for i, n_times in enumerate(list_n_times):
        fig_scaling = plt.figure(f'Scaling T={n_times}', figsize=(6, 3))
        fig_scaling.patch.set_alpha(0)
        ax_scaling = fig_scaling.subplots()
        handles = []
        xticks.append(((i + .5) * (n_bar + 1)) * width)
        labels.append(f"$T = {n_times}L$")
        for j, (strategy, name, style) in enumerate(strategies):
            col_name = ['pb', 'n_workers', 'runtime', 'runtime1']
            csv_name = (f"benchmarks_results/runtimes_n_workers_"
                        f"{n_times}_{strategy}.csv")

            try:
                df = pandas.read_csv(csv_name, names=col_name)
            except FileNotFoundError:
                print(f"Not found {csv_name}")
                continue

            runtimes_1 = df[df['n_workers'] == 1]['runtime'].values

            position = (i * (n_bar + 1) + j + 1) * width

            handles.append(ax_bar.bar(position, height=np.mean(runtimes_1),
                                      width=width, color=COLOR[j], label=name,
                                      hatch='//' if strategy == 'lgcd' else '')
                           )

            ax_bar.plot(
                np.ones_like(runtimes_1) * position,
                runtimes_1, '_', color='k')

            n_workers = df['n_workers'].unique()
            n_workers.sort()

            runtimes_scale = []
            runtimes_scale_mean = []
            for n in n_workers:
                runtimes_scale.append(df[df['n_workers'] == n].runtime.values)
                runtimes_scale_mean.append(np.mean(runtimes_scale[-1]))
            runtimes_scale_mean = np.array(runtimes_scale_mean)
            if strategy != 'random':

                t = np.logspace(0, np.log2(2 * n_workers.max()), 3, base=2)
                R0 = runtimes_scale_mean.max()

                # Linear and quadratic lines
                p = 1 if strategy == 'lgcd' else 2
                ax_scaling.plot(t, R0 / t ** p, 'k--', linewidth=1)
                tt = 2
                bbox = None  # dict(facecolor="white", edgecolor="white")
                if strategy == 'lgcd':
                    ax_scaling.text(tt, 1.4 * R0 / tt, "linear", rotation=-14,
                                    bbox=bbox, fontsize=12)
                    name_ = "DiCoDiLe-$Z$"
                else:
                    ax_scaling.text(tt, 1.4 * R0 / tt**2, "quadratic",
                                    rotation=-25, bbox=bbox, fontsize=12)
                    name_ = "DICOD"
                ax_scaling.plot(n_workers, runtimes_scale_mean, style,
                                label=name_, zorder=10, markersize=8)
                # for i, n in enumerate(n_workers):
                #     x = np.array(runtimes_scale[i])
                #     ax_scaling.plot(np.ones(value.shape) * n, value, 'k_')

        if n_times == 150:
            y_lim = (.5, 1e3)
        else:
            y_lim = (2, 2e4)
        ax_scaling.vlines(n_times / 4, *y_lim, 'g', '-.')
        ax_scaling.set_ylim(y_lim)
        ax_scaling.set_xscale('log')
        ax_scaling.set_yscale('log')
        ax_scaling.set_xlim((1, 75))
        ax_scaling.grid(True, which='both', axis='x', alpha=.5)
        ax_scaling.grid(True, which='major', axis='y', alpha=.5)
        # ax_scaling.set_xticks(n_workers)
        # ax_scaling.set_xticklabels(n_workers, fontsize=12)
        ax_scaling.set_ylabel("Runtime [sec]", fontsize=12)
        ax_scaling.set_xlabel("# workers $W$", fontsize=12)
        ax_scaling.legend(fontsize=14)
        fig_scaling.tight_layout()
        fig_scaling.savefig(f"benchmarks_results/scaling_T{n_times}.pdf",
                            dpi=300, bbox_inches='tight', pad_inches=0)

    ax_bar.set_ylabel("Runtime [sec]", fontsize=12)
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(xticks)
    ax_bar.set_xticklabels(labels, fontsize=12)
    ax_bar.set_ylim(1, 2e4)
    ax_bar.legend(bbox_to_anchor=(-.02, 1.02, 1., .3), loc="lower left",
                  handles=handles, ncol=3, fontsize=14, borderaxespad=0.)
    fig.tight_layout()
    fig.savefig("benchmarks_results/CD_strategies_comparison.png", dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.show()


def quick_plot():
    full_df = pandas.read_pickle("benchmarks_results/scaling_1d.pkl")
    list_n_times = full_df.n_times.unique()
    for T in list_n_times:
        T_df = full_df[full_df.n_times == T]
        plt.figure()
        list_reg = T_df.reg.unique()
        for reg in list_reg:
            df = T_df[T_df.reg == reg]
            # T = full_df['T'].unique()

            curve_gcd = df[df.strategy == 'greedy'].groupby('n_workers').mean()
            plt.loglog(curve_gcd.index, curve_gcd.runtime, 'C0')
            plt.loglog(curve_gcd.index, curve_gcd.t_run, 'C0--')
            curve_lgcd = df[df.strategy == 'lgcd'].groupby('n_workers').mean()
            plt.loglog(curve_lgcd.index, curve_lgcd.runtime, 'C1')
            plt.loglog(curve_lgcd.index, curve_lgcd.t_run, 'C1--')
            for strategy in ['cyclic']:  # ['cyclic', 'random']:
                curve = df[df.strategy == strategy].groupby('n_workers').mean()
                plt.loglog(curve.index, curve.runtime, 'C1')
                plt.loglog(curve.index, curve.t_run, 'C1--')
            # plt.loglog(t, t_random)

            t = df.n_workers.unique()
            plt.plot(t, np.max(curve_gcd.runtime) / t ** 2, 'k--')
            plt.plot(t, np.max(curve_lgcd.runtime) / t, 'k--')
        plt.xlim(t.min(), t.max())
        plt.savefig(f'test_{T}.pdf')
        plt.close('all')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the results of the benchmark')
    parser.add_argument('--qp', action="store_true",
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
    # list_reg = [5e-1]
    # list_n_times = [151, 750]
    list_n_times = [201, 500, 1000]

    if args.plot:
        plot_scaling_1d_benchmark(strategies, list_n_times)
    elif args.qp:
        quick_plot()
    else:
        run_scaling_1d_benchmark(
            strategies, n_rep=args.n_rep, max_workers=args.max_workers,
            soft_lock=soft_lock, list_n_times=list_n_times, list_reg=list_reg,
            random_state=random_state)
