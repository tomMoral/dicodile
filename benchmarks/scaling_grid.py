import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicodile import dicod
from dicodile.data import get_mandril
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.dictionary import init_dictionary


from joblib import Memory
mem = Memory(location='.')

ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_jobs', 'grid', 'tol', 'random_state',
    'sparsity', 'pobj'])


@mem.cache(ignore=['verbose'])
def run_one_grid(n_atoms, atom_support, reg, n_jobs, grid, tol, random_state,
                 verbose):
    # Generate a problem
    X = get_mandril()
    D = init_dictionary(X, n_atoms, atom_support, random_state=random_state)
    reg_ = reg * get_lambda_max(X, D).max()

    if grid:
        w_world = 'auto'
    else:
        w_world = n_jobs

    dicod_kwargs = dict(z_positive=False, soft_lock='corner', timeout=None,
                        max_iter=int(1e8))
    z_hat, *_, pobj, cost = dicod(
        X, D, reg=reg_, n_seg='auto', strategy='greedy', w_world=w_world,
        n_jobs=n_jobs, timing=True, tol=tol, verbose=verbose, **dicod_kwargs)

    sparsity = len(z_hat.nonzero()[0]) / z_hat.size

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      n_jobs=n_jobs, grid=grid, tol=tol,
                      random_state=random_state, sparsity=sparsity, pobj=pobj)


def run_scaling_grid(n_rep=1):
    tol = 5e-3
    n_atoms = 5
    atom_support = (8, 8)

    reg_list = np.logspace(-3, np.log10(.5), 10)[::-1][2:3]
    list_n_jobs = [1, 4, 9, 16, 25, 30, 49, 64, 100, 225]

    results = []
    for reg in reg_list:
        for grid in [True, False]:
            for n_jobs in list_n_jobs:
                if grid and n_jobs == 30:
                    n_jobs = 36
                for random_state in range(n_rep):
                    try:
                        args = (n_atoms, atom_support, reg,
                                n_jobs, grid, tol, random_state, 1)
                        res = run_one_grid(*args)
                        results.append(res)
                    except ValueError as e:
                        print(e)
                        continue

    df = pandas.DataFrame(results)
    df.to_pickle("benchmarks_results/scaling_grid.pkl")


def plot_scaling_benchmark():
    df = pandas.read_pickle("benchmarks_results/scaling_grid.pkl")

    fig = plt.figure(figsize=(6, 3))
    fig.patch.set_alpha(0)
    for name, use_grid in [("Linear Split", False),
                           ("Grid Split", True)]:
        curve = []
        res = df[df['grid'] == use_grid]
        n_jobs = res['n_jobs'].unique()
        for n in n_jobs:
            this_res = res[res['n_jobs'] == n]
            runtimes = [p[-1][1] for p in this_res['pobj']]
            curve.append((n, np.mean(runtimes)))
        curve = np.array(curve).T
        plt.semilogx(curve[0], curve[1], label=name)

    ylim = (0, 250)
    plt.vlines(512 / (8 * 4), *ylim, colors='g', linestyles='-.')
    plt.ylim(ylim)
    plt.legend(fontsize=14)
    # plt.xticks(n_jobs, n_jobs)
    plt.grid(which='both')
    plt.xlim((1, 225))
    plt.ylabel("Runtime [sec]", fontsize=12)
    plt.xlabel("# workers $W$", fontsize=12)
    plt.tight_layout()

    fig.savefig("benchmarks_results/scaling_grid.pdf", dpi=300,
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
        run_scaling_grid(n_rep=5)
