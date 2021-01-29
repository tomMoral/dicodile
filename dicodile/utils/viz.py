import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
import matplotlib.transforms as mpl_transforms

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


def display_atom(atom, ax=None, style=None):
    """Display one atom in 1D/2D with the correct formating.

    * For 1D atom, plot all the signals of all channels.
    * For 2D atom, show the image with the correct cmap for
      grayscale image with only 1 channel.

    Parameters
    ----------
    atom: ndarray, shape (n_channels, *atom_support)
        Atom to display. This should be a 1D or 2D multivariate atom.
    ax: mpl.Axes or None
        Matplotlib axe to plot the atom.
    style: dict or None
        Style info for the atom. Can include color, linestyle, linewidth, ...
    """
    if style is None:
        style = {}
    if ax is None:
        fig, ax = plt.subplots(111)

    if atom.ndim == 2:
        ax.plot(atom.T, **style)
    elif atom.ndim == 3:
        cmap = 'gray' if atom.shape[0] == 1 else None
        atom = np.rollaxis(atom, axis=0, start=3).squeeze()
        ax.imshow(atom, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set(**style)
    else:
        raise ValueError(
            'display_atom utility can only be used for multivariate atoms in '
            f'1D or 2D. Got atom with shape {atom.shape}'
        )


def display_dictionaries(*list_D, styles=None, axes=None, filename=None):
    """Display utility for dictionaries

    Parameters
    ----------
    list_D: List of ndarray, shape (n_atoms, n_channels, *atom_support)
        Dictionaries to display in the figure.
    styles: Dict of style or None
        Style to display an atom

    """
    n_dict = len(list_D)
    D_0 = list_D[0]

    if styles is None and n_dict >= 1:
        styles = [dict(color=f'C{i}') for i in range(n_dict)]

    # compute layout
    n_atoms = D_0.shape[0]
    n_cols = max(4, int(np.sqrt(n_atoms)))
    n_rows = int(np.ceil(n_atoms / n_cols))

    if axes is None:
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_dict * n_rows,
                                 squeeze=False)
    else:
        assert axes.shape >= (n_rows*n_dict, n_cols), (
            f"axes argument should have at least shape ({n_rows*n_dict}, "
            f"{n_cols}). Got {axes.shape}."
        )
        fig = axes[0, 0].get_figure()

    used_axes = 0
    for id_ax, D in enumerate(zip(*list_D)):
        used_axes += 1
        i, j = np.unravel_index(id_ax, (n_rows, n_cols))
        for k, (dk, style) in enumerate(zip(D, styles)):
            ik = n_dict * i + k
            ax = axes[ik, j]
            display_atom(dk, ax=ax, style=style)

    # hide the unused axis
    for id_ax in range(used_axes, n_cols * n_rows):
        i, j = np.unravel_index(id_ax, (n_rows, n_cols))
        for k in range(n_dict):
            ik = n_dict * i + k
            axes[ik, j].set_axis_off()

    if filename is not None:
        fig.savefig(f'{filename}.pdf', dpi=300)

    return fig


def median_curve(times, pobj):
    """Compute the Median curve, given a list of curves and their timing.

    Parameters
    ----------
    times : list of list
        Time point associated to pobj
    pobj : list of list
        Value of the cost function at a given time.

    Returns
    -------
    t: ndarray, shape (100,)
        Time points for the curve
    median_curve: ndarray, shape (100)
        Median value for the curves
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


class RotationAwareAnnotation(mpl_text.Annotation):
    """Text along a line that self adapt to rotation in the figure.

    this class is derived from the SO answer:
    https://stackoverflow.com/questions/19907140/keeps-text-rotated-in-data-coordinate-system-after-resizing#53111799

    Parameters
    ----------
    text: str
        Text to display on the figure
    anchor_point: 2-tuple
        Position of this text in the figure. The system of coordinates used to
        translate this position is controlled by the parameter ``xycoords``.
    next_point: 2-tuple
        Another point of the curve to follow. The annotation will be written
        along the line of slope dy/dx where dx = next_pt[0] - anchor_pt[0] and
        dy = next_pt[1] - anchor_pt[1].
    ax: Artiste or None
        The Artiste in which the
    **kwargs: dict
        Key-word arguments for the Annotation. List of available kwargs:
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.annotate.html
    """

    def __init__(self, text, anchor_pt, next_pt, ax=None, **kwargs):
        # Get the Artiste to draw on
        self.ax = ax or plt.gca()

        # Save the anchor point
        self.anchor_pt = np.array(anchor_pt)[None]

        # Compute the slope of the text in data coordinate system.
        dx = next_pt[0]-anchor_pt[0]
        dy = next_pt[1]-anchor_pt[1]
        ang = np.arctan2(dy, dx)
        self.angle_data = np.rad2deg(ang)

        # Create the text objects and display it
        kwargs.update(rotation_mode=kwargs.get("rotation_mode", "anchor"))
        kwargs.update(annotation_clip=kwargs.get("annotation_clip", True))
        super().__init__(text, anchor_pt, **kwargs)
        self.set_transform(mpl_transforms.IdentityTransform())
        self.ax._add_text(self)

    def _get_rotation(self):
        return self.ax.transData.transform_angles(
            np.array((self.angle_data,)), self.anchor_pt)[0]

    def _set_rotation(self, rotation):
        pass

    _rotation = property(_get_rotation, _set_rotation)
