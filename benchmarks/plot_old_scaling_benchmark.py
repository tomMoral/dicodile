import pandas
import numpy as np
import matplotlib.pyplot as plt


COLOR = ['C2', 'C1', 'C0']
SAVE_DIR = "benchmarks_results"


def plot_scaling_1d_benchmark(strategies, list_n_times):

    # compute the width of the bars
    n_group = len(list_n_times)
    n_bar = len(strategies)
    width = 1 / ((n_bar + 1) * n_group - 1)

    fig = plt.figure('comparison CD', figsize=(6, 3.5))
    fig.patch.set_alpha(0)
    ax_bar = fig.subplots()
    xticks, labels = [], []
    for i, n_times in enumerate(list_n_times):
        fig_scaling = plt.figure(f'Scaling T={n_times}', figsize=(6, 3))
        fig_scaling.patch.set_alpha(0)
        ax_scaling = fig_scaling.subplots()
        handles = []
        xticks.append(((i + .5) * (n_bar + 1)) * width)
        labels.append(f"$T = {n_times}L$")
        for j, (strategy, name, style) in enumerate(strategies):
            col_name = ['pb', 'n_jobs', 'runtime', 'runtime1']
            csv_name = (f"benchmarks_results/runtimes_n_jobs_"
                        f"{n_times}_{strategy}.csv")

            try:
                df = pandas.read_csv(csv_name, names=col_name)
            except FileNotFoundError:
                print(f"Not found {csv_name}")
                continue

            runtimes_1 = df[df['n_jobs'] == 1]['runtime'].values

            position = (i * (n_bar + 1) + j + 1) * width

            handles.append(ax_bar.bar(position, height=np.mean(runtimes_1),
                                      width=width, color=COLOR[j], label=name,
                                      hatch='//' if strategy == 'lgcd' else '')
                           )

            ax_bar.plot(
                np.ones_like(runtimes_1) * position,
                runtimes_1, '_', color='k')

            n_jobs = df['n_jobs'].unique()
            n_jobs.sort()

            runtimes_scale = []
            runtimes_scale_mean = []
            for n in n_jobs:
                runtimes_scale.append(df[df['n_jobs'] == n]['runtime'].values)
                runtimes_scale_mean.append(np.mean(runtimes_scale[-1]))
            runtimes_scale_mean = np.array(runtimes_scale_mean)
            if strategy != 'random':

                t = np.logspace(0, np.log2(2 * n_jobs.max()), 3, base=2)
                R0 = runtimes_scale_mean.max()

                # Linear and quadratic lines
                p = 1 if strategy == 'lgcd' else 2
                ax_scaling.plot(t, R0 / t ** p, 'k--', linewidth=1)
                tt = 2
                bbox = None  # dict(facecolor="white", edgecolor="white")
                if strategy == 'lgcd':
                    ax_scaling.text(tt, 1.4 * R0 / tt, "linear", rotation=-14,
                                    bbox=bbox, fontsize=12)
                    name_ = "DiCoDiLe-$Z$"
                else:
                    ax_scaling.text(tt, 1.4 * R0 / tt**2, "quadratic",
                                    rotation=-25, bbox=bbox, fontsize=12)
                    name_ = "DICOD"
                ax_scaling.plot(n_jobs, runtimes_scale_mean, style,
                                label=name_, zorder=10, markersize=8)
                # for i, n in enumerate(n_jobs):
                #     x = np.array(runtimes_scale[i])
                #     ax_scaling.plot(np.ones(value.shape) * n, value, 'k_')

        if n_times == 150:
            y_lim = (.5, 1e3)
        else:
            y_lim = (2, 2e4)
        ax_scaling.vlines(n_times / 4, *y_lim, 'g', '-.')
        ax_scaling.set_ylim(y_lim)
        ax_scaling.set_xscale('log')
        ax_scaling.set_yscale('log')
        ax_scaling.set_xlim((1, 75))
        ax_scaling.grid(True, which='both', axis='x', alpha=.5)
        ax_scaling.grid(True, which='major', axis='y', alpha=.5)
        # ax_scaling.set_xticks(n_jobs)
        # ax_scaling.set_xticklabels(n_jobs, fontsize=12)
        ax_scaling.set_ylabel("Runtime [sec]", fontsize=12)
        ax_scaling.set_xlabel("# workers $W$", fontsize=12)
        ax_scaling.legend(fontsize=14)
        fig_scaling.tight_layout()
        fig_scaling.savefig(f"benchmarks_results/scaling_T{n_times}.pdf",
                            dpi=300, bbox_inches='tight', pad_inches=0)

    ax_bar.set_ylabel("Runtime [sec]", fontsize=12)
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(xticks)
    ax_bar.set_xticklabels(labels, fontsize=12)
    ax_bar.set_ylim(1, 2e4)
    ax_bar.legend(bbox_to_anchor=(-.02, 1.02, 1., .3), loc="lower left",
                  handles=handles, ncol=3, fontsize=14, borderaxespad=0.)
    fig.tight_layout()
    fig.savefig("benchmarks_results/CD_strategies_comparison.png", dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":

    list_n_times = [150, 750]
    strategies = [
        ('greedy', 'Greedy', 's-'),
        ('random', 'Random', "h-"),
        ('lgcd', "LGCD", 'o-')
    ]
    plot_scaling_1d_benchmark(strategies, list_n_times)