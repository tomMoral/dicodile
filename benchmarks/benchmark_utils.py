import matplotlib as mpl
from pathlib import Path


# Matplotlib config
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


def get_last_file(base_dir, pattern):
    """Return the last file in a folder that match the given pattern.

    Parameters
    ----------
    base_dir: str or Path
        Base directory to search for a matching file.
    pattern: str
        Pattern used in glob to find files.

    Returns
    -------
    fname: str, name of a matching file.
    """
    base_dir = Path(base_dir)

    return sorted(base_dir.glob(pattern),
                  key=lambda x: x.stat().st_ctime, reverse=True)[0]


def mk_legend_handles(styles, **common_style):
    """Make hanldes and labels from a list of styles.

    Parameters
    ----------
    styles: list of dict
        List of style dictionary. Each dictionary should contain a `label` key.
    **common_style: dict
        All common style options. All option can be overridden in the
        individual style.

    Returns
    -------
    handles: list of Lines corresponding to the handles of the legend.
    labels: list of str containing the labels associated with each handle.
    """
    handles = []
    labels = []
    for s in styles:
        handle_style = common_style.copy()
        handle_style.update(s)
        handles.append(mpl.lines.Line2D([0], [0], **handle_style))
        labels.append(handle_style['label'])
    return handles, labels
