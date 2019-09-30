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
        super().__init__(text, anchor_pt, **kwargs)
        self.set_transform(mpl_transforms.IdentityTransform())
        self.ax._add_text(self)

    def _get_rotation(self):
        return self.ax.transData.transform_angles(
            np.array((self.angle_data,)), self.anchor_pt)[0]

    def _set_rotation(self, rotation):
        pass

    _rotation = property(_get_rotation, _set_rotation)
