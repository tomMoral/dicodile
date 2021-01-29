import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from dicodile.utils.viz import display_dictionaries


OUTPUT_DIR = pathlib.Path('benchmarks_results')
DATA_DIR = pathlib.Path('../..') / 'data' / 'images' / 'text'


# Matplotlib config
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


# Figure config
IM_NOISE_LEVEL = 3


def style_edge_axes(ax, style):
    """Color the border of an matplotlib Axes."""
    for spine in ax.spines.values():
        spine.set(**style)


def plot_dictionary(result_file='dicodile_text.py_PAMI_20-06-29_15h35.pkl',
                    res=None, s=1):
    if res is None:
        df = pd.read_pickle(OUTPUT_DIR / result_file)
        tl_max = df['text_length'].max()  # noqa: F841
        res = df.query(
            'noise_level== @IM_NOISE_LEVEL & text_length == @tl_max'
        )
        res = res.loc[res['score_cdl_2'].idxmax()]

    # compute ordering for the dictionary
    _, j = linear_sum_assignment(res['corr_cdl'], maximize=True)
    _, j_init = linear_sum_assignment(res['corr_init'], maximize=True)
    _, j_dl = linear_sum_assignment(res['corr_dl'], maximize=True)

    # Define display elements
    display_elements = {
        'Pattern': {
            'D': res['D'],
            'style': dict(color='C3', linestyle='dotted', lw=3)
         },
        'Random Patches': {
            'D': res['D_init'][j_init],
            'style': dict(color='C2', linestyle='dotted', lw=3)
        },
        'DiCoDiLe': {
            'D': res['D_cdl'][j],
            'style': dict(color='C1', lw=3)
        },
        'Online DL': {
            'D': res['D_dl'][j_dl],
            'style': dict(color='C0', lw=3)
        }
    }

    labels = list(display_elements.keys())
    list_D = [e['D'] for e in display_elements.values()]
    styles = [e['style'] for e in display_elements.values()]

    # compute layout
    n_dict = len(list_D)
    D_0 = res['D']
    n_atoms = D_0.shape[0]
    n_cols = max(4, int(np.sqrt(n_atoms)))
    n_rows = int(np.ceil(n_atoms / n_cols))
    nr = n_rows * n_dict
    fig = plt.figure(figsize=(6.4, 6.8))
    gs = mpl.gridspec.GridSpec(
        nrows=nr + 2, ncols=n_cols,
        height_ratios=[.3, .1] + [.6 / nr] * nr
    )

    # display all the atoms
    axes = np.array([[fig.add_subplot(gs[i + 2, j])
                      for j in range(n_cols)] for i in range(nr)])
    display_dictionaries(*list_D, styles=styles, axes=axes)

    # Add a legend
    handles = [mpl.lines.Line2D([0], [0], **s) for s in styles]
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.set_axis_off()
    ax_legend.legend(handles, labels, loc='center', ncol=2,
                     bbox_to_anchor=(0, 0.5, 1, .05), fontsize=14)

    # Display the original images
    data = np.load(DATA_DIR / res['filename'])
    im = data.get('X')[190:490, 250:750]

    ax = fig.add_subplot(gs[0, :n_cols // 2])
    ax.imshow(im, cmap='gray')
    ax.set_axis_off()

    ax = fig.add_subplot(gs[0, n_cols // 2:])
    noise = IM_NOISE_LEVEL * im.std() * np.random.randn(*im.shape)
    ax.imshow(im + noise, cmap='gray')
    ax.set_axis_off()

    # Adjust plot and save figure
    plt.subplots_adjust(wspace=.1, top=.99, bottom=0.01)
    fig.savefig(OUTPUT_DIR / 'dicodile_text_dict.pdf', dpi=300)


def plot_performances(result_file='dicodile_text.py_20-06-26_13h49.pkl',
                      noise_levels=[.1, IM_NOISE_LEVEL]):
    df = pd.read_pickle(OUTPUT_DIR / result_file)

    styles = {
        'score_rand_2': dict(label='Random Normal', color='k', linestyle='--',
                             linewidth=4),
        'score_init_2': dict(label='Random Patches', color='C2',
                             linestyle='--', linewidth=4),
        'score_cdl_2': dict(label='DiCoDiLe', color='C1', linewidth=4,
                            marker='o', markersize=8),
        'score_dl_2': dict(label='Online DL', color='C0', linewidth=4,
                           marker='s', markersize=8),
    }
    print(df.iloc[0][['meta_cdl', 'meta_dl']])
    cols = list(styles.keys())
    curve = df.groupby(['noise_level', 'text_length'])[cols].mean()
    err = df.groupby(['noise_level', 'text_length'])[cols].std()

    ax = None
    fig = plt.figure(figsize=(6.4, 3.6))
    fig.subplots_adjust(left=.05, right=0.98)
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=len(noise_levels),
                               hspace=.1, height_ratios=[.2, .8])
    for i, std in enumerate(noise_levels):

        ax = fig.add_subplot(gs[1, i], sharey=ax, sharex=ax)

        handles = []

        n_pixels = pd.Series({
            150: 165991.1, 360: 366754.6,
            866: 870279.6, 2081: 2059766.6,
            5000: 4881131.4
        })

        c, e = curve.loc[std], err.loc[std]
        for col, style in styles.items():
            handles.extend(ax.semilogx(n_pixels, c[col], **style))
            ax.fill_between(n_pixels, c[col] - e[col], c[col] + e[col],
                            alpha=.2, color=style['color'])
        ax.set_title(fr'$\sigma = {std}$', fontsize=14)
        ax.set_xlabel('Image Size [Mpx]', fontsize=14)
        if i == 0:
            ax.set_ylabel(r'Recovery score $\rho$', fontsize=14)
        ax.grid(True)

    # ax.set_ylim(0.55, 1)
    ax.set_xlim(n_pixels.min(), n_pixels.max())
    x_ticks = np.array([0.2, 1, 4.8]) * 1e6
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x/1e6:.1f}' for x in x_ticks])

    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.set_axis_off()
    ax_legend.legend(handles, [h.get_label() for h in handles], ncol=2,
                     loc='center', bbox_to_anchor=(0, .95, 1, .05),
                     fontsize=14)
    # fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'dicodile_text_perf.pdf', dpi=300)


if __name__ == "__main__":
    plot_dictionary()
    plot_performances()
    plt.show()
