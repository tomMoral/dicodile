import numpy as np
from numpy.random import RandomState
from dicodile.multivariate_convolutional_coding_problem import \
    MultivariateConvolutionalCodingProblem
from sys import stdout as out


DEBUG = False


def fun_rand_problem(T, S, K, d, lmbd, noise_level, seed=None):
    rng = RandomState(seed)
    rho = K / (d * S)
    D = rng.normal(scale=10.0, size=(K, d, S))
    D = np.array(D)
    nD = np.sqrt((D * D).sum(axis=-1, keepdims=True))
    D /= nD + (nD == 0)

    Z = (rng.rand(K, (T - 1) * S + 1) < rho).astype(np.float64)
    Z *= rng.normal(scale=10, size=(K, (T - 1) * S + 1))

    X = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, Z)]).sum(axis=0)
    X += noise_level * rng.normal(size=X.shape)

    z0 = np.zeros((K, (T - 1) * S + 1))
    pb = MultivariateConvolutionalCodingProblem(
        D, X, z0=z0, lmbd=lmbd)
    return pb


def fun_rand_problem_old(T, S, K, d, lmbd, noise_level, seed=None):
    rng = RandomState(seed)
    rho = K / (d * S)
    t = np.arange(S) / S
    D = [[10 * rng.rand() * np.sin(2 * np.pi * K * rng.rand() * t +
          (0.5 - rng.rand()) * np.pi)
          for _ in range(d)]
         for _ in range(K)]
    D = np.array(D)
    nD = np.sqrt((D * D).sum(axis=-1))[:, :, np.newaxis]
    D /= nD + (nD == 0)
    Z = (rng.rand(K, (T - 1) * S + 1) < rho)
    Z *= rng.normal(scale=10, size=(K, (T - 1) * S + 1))
    # shape_z = K, (T-1)*S+1
    # Z = (rng.rand(*shape_z) < rho)*rng.normal(size=shape_z)*10

    X = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, Z)]).sum(axis=0)
    X += noise_level * rng.normal(size=X.shape)

    z0 = np.zeros((K, (T - 1) * S + 1))
    pb = MultivariateConvolutionalCodingProblem(
        D, X, z0=z0, lmbd=lmbd)
    return pb


def fun_step_problem(lmbd, N=None, K=5, same=False):
    from db_marche import Database

    db = Database()
    n_ex = N
    if n_ex is not None:
        n_ex += K
    lex = db.get_data(limit=n_ex, code='max4')

    n_ex = len(lex)
    lex_train = lex[:K]
    lex_test = lex[K:n_ex]

    D = []
    D_labels = []
    for ex in lex_train:
        f = np.random.rand() > .5
        i0 = np.random.randint(len(ex.steps_annotation[f]))
        s = ex.steps_annotation[f][i0]
        step = _whiten_sig(ex)
        step = step[f*6:(f+1)*6, s[0]:s[1]]
        D += [step + .0*np.random.normal(size=step.shape)]
        D_labels += [dict(foot=f, s=i0, meta=ex.meta, step=step)]
    l_max = np.max([d.shape[1] for d in D])
    D = [np.c_[d, np.zeros((6, l_max-d.shape[1]))] for d in D]
    D = np.array(D)
    # D = .001*np.random.normal(size=D.shape)
    # D = np.cumsum(D, axis=-1)

    pbs = []
    for ex in lex_test:
        sig_W = _whiten_sig(ex)
        pbs += [(MultivariateConvolutionalCodingProblem(
            D, sig_W[:6], lmbd=lmbd), ex, 'right')]

    # DEBUG test
    if DEBUG:
        D = []
        D_labels = []
        ex = lex[0]
        sig = _whiten_sig(ex)
        ls = ex.steps_annotation[0]
        ns = len(ls)
        I0 = np.random.choice(ns, 4, replace=False)
        for i in I0:
            s = ls[i]
            step = sig[:6, s[0]:s[1]]
            D += [step + .0*np.random.normal(size=step.shape)]
            D_labels += [dict(foot='right', s=i, meta=ex.meta, step=step)]
        l_max = np.max([d.shape[1] for d in D])
        D = [np.c_[d, np.zeros((6, l_max-d.shape[1]))] for d in D]
        D = np.array(D)

        pbs = []
        ex = lex[0]
        sig_W = _whiten_sig(ex)
        pbs += [(MultivariateConvolutionalCodingProblem(
            D, sig_W[:6], lmbd=lmbd), ex, 'right')]

    return pbs, D, D_labels


def fun_rand_problems(N=10, S=100, K=10, d=6, noise_level=1, seed=None):
    rng = RandomState(seed)
    t = np.arange(S)/S
    D = [[10*rng.rand()*np.sin(2*np.pi*K*rng.rand()*t +
                               (0.5-rng.rand())*np.pi)
          for _ in range(d)]
         for _ in range(K)]
    D = np.array(D)
    nD = np.sqrt((D*D).sum(axis=-1))[:, :, np.newaxis]
    D /= nD + (nD == 0)

    rho = .1*K/(d*S)
    pbs = []
    for n in range(N):
        out.write("\rProblem construction: {:7.2%}".format(n/N))
        out.flush()
        T = rng.randint(50, 70)
        Z = (rng.rand(K, (T-1)*S+1) < rho)*rng.rand(K, (T-1)*S+1)*10
        X = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                      for Dk, zk in zip(D, Z)]).sum(axis=0)
        X += noise_level*rng.normal(size=X.shape)

        pbs += [MultivariateConvolutionalCodingProblem(
            D, X, lmbd=.1)]
    return pbs, D


def _whiten_sig(ex):
    '''Return a signal whitten for the exercice
    '''
    sig = ex.data_sensor - ex.g_sensor
    sig_b = sig[:ex.seg_annotation[0]]
    sig_b -= sig_b.mean(axis=1)[:, None]
    L = np.linalg.cholesky(sig_b.dot(sig_b.T)/100)
    P = np.linalg.inv(L)
    return P.dot(sig)
