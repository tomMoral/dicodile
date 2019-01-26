import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicodile import dicod
from dicodile.data import get_mandril
from dicodile.utils import check_random_state
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.shape_helpers import get_valid_shape


from joblib import Memory
mem = Memory(location='.')

ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_jobs', 'n_seg', 'strategy', 'tol',
    'dicod_kwargs', 'seed', 'sparsity', 'pobj'])


def get_problem(n_atoms, atom_support, seed):
    X = get_mandril()

    rng = check_random_state(seed)

    n_channels, *sig_shape = X.shape
    valid_shape = get_valid_shape(sig_shape, atom_support)

    indices = np.c_[[rng.randint(size_ax, size=(n_atoms))
                     for size_ax in valid_shape]].T
    D = np.empty(shape=(n_atoms, n_channels, *atom_support))
    for k, pt in enumerate(indices):
        D_slice = tuple([Ellipsis] + [
            slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
        D[k] = X[D_slice]
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D*D, axis=sum_axis, keepdims=True))

    return X, D


@mem.cache(ignore=['timeout', 'max_iter', 'verbose'])
def run_one(n_atoms, atom_support, reg, n_jobs, strategy, tol, seed,
            timeout, max_iter, verbose, dicod_kwargs):
    # Generate a problem
    X, D = get_problem(n_atoms, atom_support, seed)
    lmbd = reg * get_lambda_max(X[None], D).max()

    if strategy == 'lgcd':
        n_seg = 'auto'
        effective_strategy = 'greedy'
    elif strategy in ["greedy", 'random']:
        n_seg = 1
        effective_strategy = strategy
    else:
        raise NotImplementedError(f"Bad strategy name {strategy}")

    z_hat, *_, pobj, cost = dicod(
        X, D, reg=lmbd, n_seg=n_seg, strategy=effective_strategy,
        n_jobs=n_jobs, timing=True, tol=tol, timeout=timeout,
        max_iter=max_iter, verbose=verbose, **dicod_kwargs)

    sparsity = len(z_hat.nonzero()[0]) / z_hat.size

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      n_jobs=n_jobs, n_seg=n_seg, strategy=strategy,
                      tol=tol, dicod_kwargs=dicod_kwargs, seed=seed,
                      sparsity=sparsity, pobj=pobj)


def run_scaling_benchmark(max_n_jobs, n_rep=1):
    tol = 1e-3
    n_atoms = 5
    atom_support = (8, 8)

    verbose = 1
    timeout = 9000
    max_iter = int(1e8)

    dicod_kwargs = dict(z_positive=False, soft_lock='border')

    reg_list = np.logspace(-3, np.log10(.5), 10)[::-1][:3]

    list_n_jobs = np.round(np.logspace(0, np.log10(20), 10)).astype(int)
    list_n_jobs = [int(v * v) for v in np.unique(list_n_jobs)[::-1]]

    results = []
    for reg in reg_list:
        for n_jobs in list_n_jobs:
            for strategy in ['greedy', 'lgcd']:  # , 'random']:
                for seed in range(n_rep):
                    res = run_one(n_atoms, atom_support, reg, n_jobs, strategy,
                                  tol, seed, timeout, max_iter, verbose,
                                  dicod_kwargs)
                    results.append(res)

    df = pandas.DataFrame(results)
    df.to_pickle("benchmarks_results/scaling_n_jobs.pkl")


def plot_scaling_benchmark():
    df = pandas.read_pickle("benchmarks_results/scaling_n_jobs.pkl")
    import matplotlib.lines as mlines
    handles_lmbd = {}
    handles_strat = {}
    fig = plt.figure(figsize=(6, 3))
    fig.patch.set_alpha(0)

    ax = plt.subplot()

    colors = ['C0', 'C1', 'C2']
    n_jobs = df['n_jobs'].unique()
    regs = df['reg'].unique()
    regs.sort()
    for reg, c in zip(regs, colors):
        for strategy, style in [('Greedy', '--'), ('LGCD', '-')]:
            s = strategy.lower()
            this_res = df[(df['reg'] == reg) & (df['strategy'] == s)]
            runtimes = []
            runtime_std = []
            for n in n_jobs:
                pobj = this_res[this_res['n_jobs'] == n]['pobj'].values
                end_times = [rt[-1][1] for rt in pobj if rt is not None]
                runtimes.append(np.mean(end_times))
                runtime_std.append(np.std(end_times))
            runtimes, runtime_std = np.array(runtimes), np.array(runtime_std)

            plt.loglog(n_jobs, runtimes, label=f"{strategy}_{reg:.2f}",
                       linestyle=style, c=c)
            plt.fill_between(n_jobs, runtimes - runtime_std,
                             runtimes + runtime_std, alpha=.1)
            color_handle = mlines.Line2D(
                [], [], linestyle='-', c=c, label=f"${reg:.2f}\lambda_\max$")
            style_handle = mlines.Line2D(
                [], [], linestyle=style, c='k', label=f"{strategy}")
            handles_lmbd[reg] = color_handle
            handles_strat[strategy] = style_handle
    plt.xlim((1, 400))
    # plt.ylim((1e1, 1e4))
    # plt.xticks(n_jobs, n_jobs, fontsize=14)
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
    plt.legend(handles=handles_strat.values(), loc=3, fontsize=14)
    ax.add_artist(legend_lmbd)
    plt.tight_layout()
    plt.savefig("benchmarks_results/scaling_n_jobs.pdf", dpi=300,
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
        run_scaling_benchmark(max_n_jobs=400, n_rep=5)
