
import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicodile.data import get_hubble
from dicodile.dicodile import dicodile
from dicodile.utils.viz import median_curve
from dicodile.utils.dictionary import get_lambda_max
from dicodile.utils.dictionary import init_dictionary

from benchmarks.other.sporco.dictlrn.prlcnscdl import \
    ConvBPDNDictLearn_Consensus

from joblib import Memory
mem = Memory(location='.')


ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_jobs', 'random_state', 'method',
    'z_positive', 'times', 'pobj'])


@mem.cache
def run_one(method, n_atoms, atom_support, reg, z_positive, n_jobs, n_iter,
            tol, eps, random_state):

    X = get_hubble()[:, 512:1024, 512:1024]
    D_init = init_dictionary(X, n_atoms, atom_support,
                             random_state=random_state)

    if method == 'wohlberg':
        ################################################################
        #            Run parallel consensus ADMM
        #
        lmbd_max = get_lambda_max(X, D_init).max()
        print("Lambda max = {}".format(lmbd_max))
        reg_ = reg * lmbd_max

        D_init_ = np.transpose(D_init, axes=(3, 2, 1, 0))
        X_ = np.transpose(X[None], axes=(3, 2, 1, 0))

        options = {
            'Verbose': True,
            'StatusHeader': False,
            'MaxMainIter': n_iter,
            'CCMOD': {'rho': 1.0,
                      'ZeroMean': False},
            'CBPDN': {'rho': 50.0*reg_ + 0.5,
                      'NonNegCoef': z_positive},
            'DictSize': D_init_.shape,
            }
        opt = ConvBPDNDictLearn_Consensus.Options(options)
        cdl = ConvBPDNDictLearn_Consensus(
            D_init_, X_, lmbda=reg_, nproc=n_jobs, opt=opt, dimK=1, dimN=2)

        _, pobj = cdl.solve()
        print(pobj)

        itstat = cdl.getitstat()
        times = itstat.Time

    elif method == "dicodile":
        pobj, times, D_hat, z_hat = dicodile(
            X, D_init, reg=reg, z_positive=z_positive, n_iter=n_iter, eps=eps,
            n_jobs=n_jobs, verbose=2, tol=tol)
        pobj = pobj[::2]
        times = np.cumsum(times)[::2]

    else:
        raise NotImplementedError()

    return ResultItem(
        n_atoms=n_atoms, atom_support=atom_support, reg=reg, n_jobs=n_jobs,
        random_state=random_state, method=method, z_positive=z_positive,
        times=times, pobj=pobj)


def run_benchmark(methods=['wohlberg', 'dicodile'],
                  runs=range(5)):
    n_iter = 501
    n_jobs = 36
    reg = .1
    tol = 1e-3
    eps = 1e-4
    n_atoms = 36
    atom_support = (28, 28)
    z_positive = True
    args = (n_atoms, atom_support, reg, z_positive, n_jobs)

    results = []
    # rng = check_random_state(42)

    # for method in ['wohlberg']:
    for method in methods:
        for random_state in runs:
            if method == 'dicodile':
                results.append(run_one(method, *args, 100, tol, eps,
                                       random_state))
            else:
                results.append(run_one(method, *args, n_iter, 0, 0,
                                       random_state))

    # Save results
    df = pandas.DataFrame(results)
    df.to_pickle("benchmarks_results/compare_cdl.pkl")


def plot_results():
    df = pandas.read_pickle("benchmarks_results/compare_cdl.pkl")

    fig = plt.figure("compare_cdl", figsize=(6, 3))
    fig.patch.set_alpha(0)
    tt, tt_w = [], []
    pp, pp_w = [], []
    for i in range(5):
        times_w = df[df['method'] == 'wohlberg']['times'].values[i]
        pobjs_w = np.array(df[df['method'] == 'wohlberg']['pobj'].values[i])
        times = df[df['method'] == 'dicodile']['times'].values[i]
        pobjs = np.array(df[df['method'] == 'dicodile']['pobj'].values[i])
        tt.append(times)
        pp.append(pobjs)
        tt_w.append(times_w)
        pp_w.append(pobjs_w)
        times = np.r_[.1, times[1:]]
        times_w = np.r_[.1, times_w[1:]]
        plt.plot(times, (pobjs), 'C0', alpha=.3)
        plt.plot(times_w, (pobjs_w[1:]), 'C1--', alpha=.3)
    tt, pp = median_curve(tt, pp)
    print(tt)
    tt_w, pp_w = median_curve(tt_w, pp_w)
    plt.semilogy(tt_w, pp_w, 'C1', label='Skau et al. (2018)')
    plt.semilogy(tt, pp, 'C0', label='DiCoDiLe')
    plt.legend(fontsize=14)
    plt.xlabel("Time [sec]", fontsize=12)
    plt.ylabel("$F(Z, D) / F(0, 0)$", fontsize=12)
    plt.tight_layout()
    fig.savefig("benchmarks_results/compare_cdl.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Compare DICODIL with wohlberg')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the result of the benchmark')
    parser.add_argument('--method', type=str, default=None,
                        help='Only run one method and one run')
    parser.add_argument('--run', type=int, default=0,
                        help='PB to run run')
    parser.add_argument('--n_rep', type=int, default=5,
                        help='Number of repetition')
    args = parser.parse_args()

    if args.plot:
        plot_results()
    elif args.method is not None:
        run_benchmark(methods=[args.method], runs=[args.run])
    else:
        run_benchmark(runs=range(args.n_rep))
