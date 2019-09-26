import pandas
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicodile.update_z import dicod
from dicodile.data.images import get_mandril
from dicodile.utils import check_random_state
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.shape_helpers import get_valid_support


from joblib import Memory
mem = Memory(location='.')

ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_workers', 'n_seg', 'strategy', 'tol',
    'dicod_kwargs', 'random_state', 'sparsity', 'iterations', 'runtime',
    't_init', 't_run', 'n_updates', 't_select', 't_update'])

RESULT_DIR = pathlib.Path("benchmark_results")
PKL_FILENAME = RESULT_DIR / "scaling_n_workers.pkl"


def get_problem(n_atoms, atom_support, random_state):
    X = get_mandril()

    rng = check_random_state(random_state)

    n_channels, *sig_support = X.shape
    valid_support = get_valid_support(sig_support, atom_support)

    indices = np.c_[[rng.randint(size_ax, size=(n_atoms))
                     for size_ax in valid_support]].T
    D = np.empty(shape=(n_atoms, n_channels, *atom_support))
    for k, pt in enumerate(indices):
        D_slice = tuple([Ellipsis] + [
            slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
        D[k] = X[D_slice]
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D*D, axis=sum_axis, keepdims=True))

    return X, D


@mem.cache(ignore=['timeout', 'max_iter', 'verbose'])
def run_one(n_atoms, atom_support, reg, n_workers, strategy, tol, random_state,
            timeout, max_iter, verbose, dicod_kwargs):
    # Generate a problem
    X, D = get_problem(n_atoms, atom_support, random_state)
    lmbd = reg * get_lambda_max(X[None], D).max()

    if strategy == 'lgcd':
        n_seg = 'auto'
        effective_strategy = 'greedy'
    elif strategy in ["greedy", 'random']:
        n_seg = 1
        effective_strategy = strategy
    else:
        raise NotImplementedError(f"Bad strategy name {strategy}")

    z_hat, *_, run_statistics = dicod(
        X, D, reg=lmbd, n_seg=n_seg, strategy=effective_strategy,
        n_workers=n_workers, timing=True, tol=tol, timeout=timeout,
        max_iter=max_iter, verbose=verbose, **dicod_kwargs)

    sparsity = len(z_hat.nonzero()[0]) / z_hat.size

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      n_workers=n_workers, n_seg=n_seg, strategy=strategy,
                      tol=tol, dicod_kwargs=dicod_kwargs, sparsity=sparsity,
                      random_state=random_state, **run_statistics)


def run_scaling_benchmark(max_n_workers, n_rep=1):
    tol = 1e-3
    n_atoms = 5
    atom_support = (8, 8)

    verbose = 1
    timeout = 9000
    max_iter = int(1e8)

    dicod_kwargs = dict(z_positive=False, soft_lock='border')

    reg_list = np.logspace(-3, np.log10(.5), 10)[::-1][:3]

    list_n_workers = np.round(np.logspace(0, np.log10(20), 10)).astype(int)
    list_n_workers = [int(v * v) for v in np.unique(list_n_workers)[::-1]]

    results = []
    for reg in reg_list:
        for n_workers in list_n_workers:
            for strategy in ['greedy', 'lgcd']:  # , 'random']:
                for random_state in range(n_rep):
                    res = run_one(
                        n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                        n_workers=n_workers, strategy=strategy, tol=tol,
                        timeout=timeout, max_iter=max_iter, verbose=verbose,
                        dicod_kwargs=dicod_kwargs, random_state=random_state)
                    results.append(res)

    df = pandas.DataFrame(results)
    df.to_pickle(PKL_FILENAME)


def plot_scaling_benchmark():
    df = pandas.read_pickle(PKL_FILENAME)
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
        for strategy, style in [('Greedy', '--'), ('LGCD', '-')]:
            s = strategy.lower()
            this_df = df[(df.reg == reg) & (df.strategy == s)]
            curve = this_df.groupby('n_workers').runtime
            runtimes = curve.mean()
            runtime_std = curve.std()

            plt.fill_between(runtimes.index, runtimes - runtime_std,
                             runtimes + runtime_std, alpha=.1)
            plt.loglog(runtimes.index, runtimes, label=f"{strategy}_{reg:.2f}",
                       linestyle=style, c=c)
            color_handle = lines.Line2D(
                [], [], linestyle='-', c=c, label=f"${reg:.2f}\\lambda_\\max$")
            style_handle = lines.Line2D(
                [], [], linestyle=style, c='k', label=f"{strategy}")
            handles_lmbd[reg] = color_handle
            handles_strategy[strategy] = style_handle
    plt.xlim((1, 400))
    # plt.ylim((1e1, 1e4))
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
    plt.legend(handles=handles_strategy.values(), loc=3, fontsize=14)
    ax.add_artist(legend_lmbd)
    plt.tight_layout()
    plt.savefig("benchmarks_results/scaling_n_workers.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Benchmark scaling performance for DICOD')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the result of the benchmark')
    args = parser.parse_args()

    if args.plot:
        plot_scaling_benchmark()
    else:
        run_scaling_benchmark(max_n_workers=400, n_rep=5)
