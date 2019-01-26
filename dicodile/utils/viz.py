import numpy as np
import matplotlib.pyplot as plt

from .csc import reconstruct


def plot_atom_and_coefs(D_hat, z_hat, prefix):
    n_atoms = D_hat.shape[0]

    E = np.sum(z_hat > 0, axis=(1, 2))
    i0 = E.argsort()[::-1]

    n_cols = 5
    n_rows = int(np.ceil(n_atoms / n_cols))
    fig = plt.figure(figsize=(3*n_cols + 2, 3*n_rows + 2))
    fig.patch.set_alpha(0)
    for i in range(n_rows):
        for j in range(n_cols):
            if n_cols * i + j >= n_atoms:
                continue
            k = i0[n_cols * i + j]
            ax = plt.subplot2grid((n_rows, n_cols), (i, j))
            scale = 1 / D_hat[k].max() * .99
            Dk = np.clip(scale * D_hat[k].swapaxes(0, 2), 0, 1)
            ax.imshow(Dk)
            ax.axis('off')
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=.1, hspace=.1)

    fig.savefig(f"hubble/{prefix}dict.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0)

    fig = plt.figure()
    fig.patch.set_alpha(0)
    plt.imshow(z_hat.sum(axis=0).T > 0, cmap='gray')
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(f"hubble/{prefix}z_hat.pdf", dpi=1200,
                bbox_inches='tight', pad_inches=0)

    fig = plt.figure()
    fig.patch.set_alpha(0)
    X_hat = np.clip(reconstruct(z_hat, D_hat), 0, 1)
    plt.imshow(X_hat.swapaxes(0, 2))
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(f"hubble/{prefix}X_hat.pdf", dpi=1200,
                bbox_inches='tight', pad_inches=0)


def median_curve(times, pobj):
    """Compute the Median curve, given a list of curves and their timing.

    times : list of list
        Time point associated to pobj
    pobj : list of list
        Value of the cost function at a given time.
    """
    T = np.max([np.max(tt) for tt in times])
    # t = np.linspace(0, T, 100)
    t = np.logspace(-1, np.log10(T), 100)
    curves = []
    for lt, lf in zip(times, pobj):
        curve = []
        for tt in t:
            i0 = np.argmax(lt > tt)
            if i0 == 0 and tt != 0:
                value = lf[-1]
            elif i0 == 0 and tt == 0:
                value = lf[0]
            else:
                value = (lf[i0] - lf[i0-1]) / (lt[i0] - lt[i0-1]) * (
                    tt - lt[i0-1]) + lf[i0-1]
            curve.append(value)
        curves.append(curve)
    return t, np.median(curves, axis=0)
