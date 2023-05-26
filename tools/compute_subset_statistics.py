"""
This script computes the statistics for the CausalOrca dataset, including:
- The number of scenes
- The number of agents
- The number of scenes with a given number of agents
- The number of direct causal agents, indirect causal agents, and non-causal agents


"""

import os
import pathlib
from collections import Counter

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style("ticks")
sns.set_palette("flare")

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)
plt.rcParams.update({
    # 'figure.figsize': (15, 5),
    'figure.titlesize': '28',
    'axes.titlesize': '22',
    'legend.title_fontsize': '16',
    'legend.fontsize': '14',
    'axes.labelsize': '18',
    'xtick.labelsize': '16',
    'ytick.labelsize': '16',
    'figure.dpi': 200,
})

SHOW_PLOTS = True
BINWIDTH = 0.072
HEIGHT = 7.8
NC_THRESHOLD = 0.02
C_THRESHOLD = 0.1

import argparse
import pickle

import numpy as np


def _compute_ade(traj1, traj2):
    return np.mean(np.linalg.norm(traj1 - traj2, axis=-1), axis=-1).item()


def _compute_min_distance(traj1, traj2):
    return np.min(np.linalg.norm(traj1 - traj2, axis=-1), axis=-1).item()


def compute_revised_agent_type(direct_causal: bool, causal_efffect: float):
    # ┌──────┬────────────────┬─────────────────────────────────────┐
    # │      │                │                                     │
    # │  N   │  NA            │  Direct   Causal                    │
    # │  o   │                │                                     │
    # │  n   │                │                                     │
    # │  -   │                │                                     │
    # │  c   │                │                                     │
    # │  a   │                │                                     │
    # │  u   │                │                                     │
    # │  s   │                ├─────────────────────────────────────┤
    # │  a   │                │                                     │
    # │  l   │                │  Indirect Causal                    │
    # │      │                │                                     │
    # └──────┴────────────────┴─────────────────────────────────────┴────►
    #       0.02             0.1                                       Causal Effect
    assert causal_efffect >= 0

    if causal_efffect <= NC_THRESHOLD:
        return "non-causal"

    if causal_efffect > NC_THRESHOLD and causal_efffect <= C_THRESHOLD:
        return "na"

    if direct_causal:
        return "direct causal"

    return "indirect causal"


def add_more_metrics_to_dataset(dataset, ego_agent_idx=0, future_start_idx=8):
    for scene in dataset["scenes"]:
        n_agents, n_timesteps, _ = scene["trajectories"].shape
        assert scene["trajectories"].shape == (n_agents, n_timesteps, 2)
        assert scene["remove_agent_i_trajectories"].shape == (n_agents, n_agents, n_timesteps, 2)

        causal_effect = []
        causal_effect_future = []
        min_distance_to_ego_future = []
        direct_causal = []
        revised_agent_type = []
        for removed_agent_idx in range(n_agents):

            if removed_agent_idx == ego_agent_idx:
                continue

            # The causal effect for all timesteps (past and future)
            factual_ego_trajectory = scene["trajectories"][ego_agent_idx, :, :]
            counterfactual_ego_trajectory = scene["remove_agent_i_trajectories"][removed_agent_idx, ego_agent_idx, :, :]
            ce_i = _compute_ade(factual_ego_trajectory, counterfactual_ego_trajectory)

            # The causal effect for the future
            factual_ego_future_trajectory = factual_ego_trajectory[future_start_idx:, :]
            counterfactual_ego_future_trajectory = counterfactual_ego_trajectory[future_start_idx:, :]
            cef_i = _compute_ade(factual_ego_future_trajectory, counterfactual_ego_future_trajectory)

            # The min distance to the ego for the future
            factual_removed_agent_future_trajectory = scene["trajectories"][removed_agent_idx, future_start_idx:, :]
            min_distance_to_ego_future += [
                _compute_min_distance(factual_ego_future_trajectory, factual_removed_agent_future_trajectory)
            ]

            # Compute direct causality labels as to be able to compute the revised agent type
            dc_i = scene["causality_labels"][ego_agent_idx, :, removed_agent_idx].any()

            # The revised agent type
            rat_i = compute_revised_agent_type(dc_i, cef_i)

            causal_effect += [ce_i]
            causal_effect_future += [cef_i]
            direct_causal += [dc_i]
            revised_agent_type += [rat_i]

        non_ego_indices = [i for i in range(n_agents) if i != ego_agent_idx]
        precomputed_causal_effect = scene["remove_agent_i_ade"][non_ego_indices, ego_agent_idx]
        recomputed_causal_effect = np.array(causal_effect)
        assert np.allclose(precomputed_causal_effect, recomputed_causal_effect)

        scene["causal_effect_future"] = np.array(causal_effect_future)
        scene["min_distance_to_ego_future"] = np.array(min_distance_to_ego_future)
        scene["direct_causal"] = np.array(direct_causal)
        scene["revised_agent_type"] = np.array(revised_agent_type)


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def jointplot_min_distance_to_ego_wrt_causal_effect(df, plot_id="min_distance_to_causal_effect"):
    jointgrid = sns.jointplot(
        data=df,
        y="min distance to ego (future)",
        x="causal effect (future)",
        xlim=(0, 4),
        ylim=(0, 10),
        kind="hist",
        height=HEIGHT,
        ratio=5,
        binwidth=(BINWIDTH, BINWIDTH * 10 / 4),
        vmin=None, vmax=None,
        cmap="flare",
        cbar=True,
        marginal_kws={"binwidth": BINWIDTH},
        norm=LogNorm(),
    )
    jointgrid.figure.subplots_adjust(left=0.12, right=0.87, top=0.86, bottom=0.13)
    pos_joint_ax = jointgrid.ax_joint.get_position()
    pos_marg_x_ax = jointgrid.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    jointgrid.ax_joint.set_position(
        [pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    jointgrid.fig.axes[-1].set_position([.87, pos_joint_ax.y0, .07, pos_joint_ax.height])

    jointgrid.figure.suptitle("Min distance to ego wrt causal effect")
    # jointgrid.set_axis_labels(xlabel=rf"distance to ego", ylabel=rf"causal effect")

    plt.savefig(os.path.join(args.plots_path, f"{plot_id}.png"))
    plt.savefig(os.path.join(args.plots_path, f"PDF__{plot_id}.pdf"))
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def histplot_causal_effect(df, plot_id="causal_effect_histogram"):
    for logy in [False, True]:
        sns.histplot(
            data=df,
            x="causal effect (future)",
            binwidth=BINWIDTH,
            log_scale=(False, logy),
        )
        plt.title("Causal effect distribution (log scale)" if logy else "Causal effect distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_path, f"{plot_id}__logy.png" if logy else f"{plot_id}.png"))
        plt.savefig(os.path.join(args.plots_path, f"PDF__{plot_id}__logy.pdf" if logy else f"PDF__{plot_id}.pdf"))
        if SHOW_PLOTS:
            plt.show()
        plt.close()


def histplot_min_distance_to_ego_per_agent_type(df, plot_id="min_distance_to_ego_histogram"):
    df = df[df["revised agent type"] != "na"]
    for logy in False, True:
        facet_grid = sns.displot(
            data=df,
            x="min distance to ego (future)",
            hue="revised agent type",
            hue_order=["direct causal", "indirect causal", "non-causal"],
            row="revised agent type",
            row_order=["direct causal", "indirect causal", "non-causal"],
            binwidth=BINWIDTH,
            stat="probability",
            common_norm=False,
            palette="flare",
            height=HEIGHT / 3,
            aspect=2.8,
            log_scale=(False, logy),
        )
        # facet_grid.fig.subplots_adjust(top=0.9)
        facet_grid.fig.suptitle("Min distance to ego per agent type" + (" (log scale)" if logy else ""))
        plt.xlim(0, 10)
        plt.gca().set_ylabel("probability")
        facet_grid.set_titles(row_template="")
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_path, f"{plot_id}{'__logy' if logy else ''}.png"))
        plt.savefig(os.path.join(args.plots_path, f"PDF__{plot_id}{'__logy' if logy else ''}.pdf"))
        if SHOW_PLOTS:
            plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pickle_path', type=str, required=True,
                        help='Path to dataset pickle file')
    parser.add_argument('--no_latex', action='store_true',
                        help='Whether not to use latex when creating plots.'
                             ' Some systems might not have a tex distribution installed.')
    parser.add_argument('--plots_path', type=str, default="plots",
                        help='Path to save plots to')
    args = parser.parse_args()
    print(f"args={args}")

    print(f"Saving plots to {os.path.abspath(args.plots_path)}")
    ensure_dir(args.plots_path)
    print()

    with open(args.data_pickle_path, "rb") as f:
        dataset = pickle.load(f)

    add_more_metrics_to_dataset(dataset)


    def flatten_metric(metric_name):
        return np.array([value for scene in dataset["scenes"] for value in scene[metric_name]])


    df = pd.DataFrame({
        "causal effect (future)": flatten_metric("causal_effect_future"),
        "min distance to ego (future)": flatten_metric("min_distance_to_ego_future"),
        "revised agent type": flatten_metric("revised_agent_type"),
    })

    print("Number of scenes:", len(dataset["scenes"]))
    print("Number of agents:", len(df))
    print("Average number of agents per scene:", len(df) / len(dataset["scenes"]))
    print("Number of agents per scene distribution:")
    counter = Counter([len(scene["trajectories"]) for scene in dataset["scenes"]])
    for num_agents, num_scenes in sorted(counter.items(), key=lambda x: x[0]):
        print(f"{num_agents}: {100 * num_scenes / len(dataset['scenes']):.1f}% ({num_scenes})")
    print()

    print("Agent ratios (for revised agent type):")
    for revised_agent_type in df["revised agent type"].unique():
        print(f"{revised_agent_type}: "
              f"{100 * len(df[df['revised agent type'] == revised_agent_type]) / len(df):.1f}% "
              f"({len(df[df['revised agent type'] == revised_agent_type])})")
    print()

    print("The average minimum distance to ego:", df["min distance to ego (future)"].mean())
    print("The average minimum distance to ego per agent type:")
    for revised_agent_type in df["revised agent type"].unique():
        print(f"{revised_agent_type}: "
              f"{df[df['revised agent type'] == revised_agent_type]['min distance to ego (future)'].mean():.4f}")
    print()

    # Jointplot of min distance wrt causal effect
    jointplot_min_distance_to_ego_wrt_causal_effect(df)
    for revised_agent_type in df["revised agent type"].unique():
        jointplot_min_distance_to_ego_wrt_causal_effect(
            df[df["revised agent type"] == revised_agent_type],
            plot_id=f"min_distance_to_ego_wrt_causal_effect__{revised_agent_type}",
        )

    # Plot a histogram of the causal effect
    histplot_causal_effect(df)

    # Plot a histogram of the min distance to ego, per agent type
    histplot_min_distance_to_ego_per_agent_type(df)

    print("Done.")
