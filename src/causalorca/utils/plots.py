import math
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from causalorca.orca.constants import compute_agent_types
from causalorca.utils.util import ensure_dir
from causalorca.utils.visualize_as_animation import animate


def setup_latex_for_matplotlib():
    from matplotlib import pyplot as plt
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)


def plot_trajectory(scene, figname='viz.png', lim=6.0):
    '''
        trajectories: list of trajectories in the scene
    '''
    figname_folder = os.path.dirname(figname)
    ensure_dir(figname_folder)

    num_traj = scene.shape[0]
    num_frame = scene.shape[1]
    cm_subsection = np.linspace(0.0, 1.0, num_traj)
    colors = [matplotlib.cm.jet(x) for x in cm_subsection]

    for i in range(num_traj):
        for k in range(1, num_frame):
            alpha = 1.0 - k / num_frame
            width = (1.0 - alpha) * 15.0
            plt.plot(scene[i, k - 1:k + 1, 0], scene[i, k - 1:k + 1, 1],
                     '-o', color=colors[i], alpha=alpha, linewidth=width, label=None if k > 1 else f"A{i + 1:02d}")

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(figname)
    plt.close()


def plot_with_agent_removal(orca_result, logs_path, ego_agent_idx=0, indirect_causal_ade_threshold=0.0,
                            plot_id="agent_removal", plot_title="Agent Removal Trajectories", usetex=True):
    # Import and set up plotting libraries
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_style("ticks")
    sns.set_palette("deep")

    plt.rc('font', **{'size': 12})
    plt.rc('figure', dpi=100)
    if usetex:
        setup_latex_for_matplotlib()

    # Prepare data
    n_agents = len(orca_result["trajectories"])
    agent_type_shorthand = orca_result["agent_type_shorthand"]

    # Trajectory plotting help function
    def _plot_trajectory_to_axis(trajectory, ax, agent_idx_removed=None):
        trajectory_agent_df_list = []
        for agent_idx, type in enumerate(agent_type_shorthand):
            df = pd.DataFrame({
                "x": trajectory[agent_idx][:, 0],
                "y": trajectory[agent_idx][:, 1],
            })
            agent_idx_str = f"A{agent_idx + 1:02d} ({type})"
            if agent_idx_removed is not None and agent_idx == agent_idx_removed:
                agent_idx_str = "NO " + agent_idx_str

            df["agent idx"] = agent_idx_str
            df["agent type"] = type
            assert len(trajectory[agent_idx]) == 20
            df["timestep type"] = ["past"] * 8 + ["future"] * 12
            trajectory_agent_df_list += [df]

        # Add old ego agent
        if agent_idx_removed is not None and agent_idx_removed != ego_agent_idx:
            df = pd.DataFrame({
                "x": orca_result["trajectories"][ego_agent_idx][:, 0],
                "y": orca_result["trajectories"][ego_agent_idx][:, 1],
            })
            df["agent idx"] = f"Original A{ego_agent_idx + 1:02d} (EGO)"
            df["agent type"] = "EGO"
            df["timestep type"] = ["past"] * 8 + ["future"] * 12
            trajectory_agent_df_list += [df]

        trajectory_df = pd.concat(trajectory_agent_df_list, ignore_index=True)

        sns.lineplot(data=trajectory_df, x="x", y="y", hue="agent idx", style="timestep type",
                     alpha=0.6, markers=True, dashes=False, palette="deep", sort=False, ax=ax)

    # Start plotting
    n_cols = 3
    n_rows = 1 + math.ceil(n_agents / 3)
    fig = plt.figure(constrained_layout=True, figsize=(6.4 * n_cols, 4.8 * n_rows))
    fig.suptitle(plot_title, fontsize=24)
    subfigs = fig.subfigures(n_rows, n_cols, wspace=0.07, )  # width_ratios=[2,1]

    # [0,0] - Original trajectory
    subfig, title = subfigs[0, 0], f"Original trajectory ({'valid' if orca_result['validity'] else 'invalid'})"
    subfig.suptitle(title)  # fontsize=18
    ax = subfig.subplots()
    _plot_trajectory_to_axis(orca_result["trajectories"], ax)
    original_trajectory_ax = ax

    # [0,1] - Causality Effect estimated using ADE
    subfig, title = subfigs[0, 1], "Causality Effect (estimated using ADE)"
    subfig.suptitle(title)
    ax = subfig.subplots()
    agent_idx_list = [f"A{agent_idx + 1:02d} ({type})" for agent_idx, type in enumerate(agent_type_shorthand)]
    ade_heatmap_df = pd.DataFrame(orca_result["remove_agent_i_ade"], columns=agent_idx_list, index=agent_idx_list)
    sns.heatmap(ade_heatmap_df, annot=True, linewidth=.5, ax=ax, fmt='.2g')

    # [0,2] - Indirect Causal Effect based on the simulator causality graph
    subfig = subfigs[0, 2]
    title1 = 'Direct Causal Effect ("causality graph"-based)'
    title2 = 'Indirect Causal Effect ("causality graph"-based)'

    axes = subfig.subplots(2, 1)
    agent_idx_list = [f"A{agent_idx + 1:02d}" for agent_idx in range(n_agents)]

    ax = axes[0]
    ax.title.set_text(title1)
    causal_at_any_timestep_from_to = orca_result["causality_labels"].any(axis=-2).T
    directly_causal_heatmap_df = pd.DataFrame(
        causal_at_any_timestep_from_to.astype(bool),
        columns=agent_idx_list,
        index=agent_idx_list,
    )
    sns.heatmap(directly_causal_heatmap_df, annot=True, linewidth=.5, vmin=0, vmax=1, ax=ax)

    ax = axes[1]
    ax.title.set_text(title2)
    ice_heatmap_df = pd.DataFrame(
        orca_result["indirect_causality_effect"].astype(bool),
        columns=agent_idx_list,
        index=agent_idx_list,
    )
    sns.heatmap(ice_heatmap_df, annot=True, linewidth=.5, vmin=0, vmax=1, ax=ax, fmt='.2g')

    # [1+, ...] - Trajectories when each of the agents gets removed
    for agent_idx in range(n_agents):
        validity_str = 'valid' if orca_result['remove_agent_i_validity'][agent_idx] else 'invalid'
        title = f'Trajectory w/ agent A{agent_idx + 1:02d} removed ({validity_str})'
        subfig = subfigs[1 + agent_idx // 3, agent_idx % 3]
        subfig.suptitle(title)
        ax = subfig.subplots()
        _plot_trajectory_to_axis(orca_result["remove_agent_i_trajectories"][agent_idx], ax, agent_idx)
        ax.sharex(original_trajectory_ax)
        ax.sharey(original_trajectory_ax)

    print("AR plot saving...")
    plt.savefig(os.path.join(logs_path, f"{plot_id} -- {plot_title}.png"))
    # plt.savefig(os.path.join(logs_path, f"PDF {plot_id} -- {plot_title}.pdf"))
    print("Saved!")
    plt.close()


def plot_causality_effect_per_agent_type(
        scenes, logs_path, ego_agent_idx=0,
        plot_id="ceffect", plot_title="Causality Effect (estimated by agent removal ADE)", usetex=True,
):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_style("ticks")
    sns.set_palette("flare")
    # sns.set_palette("deep")

    plt.rc('font', **{'size': 12})
    plt.rc('figure', dpi=200)
    if usetex:
        setup_latex_for_matplotlib()

    causality_effect_df = pd.DataFrame({
        "agent type": [
            agent_type
            for scene in scenes
            for agent_idx, agent_type in enumerate(scene["agent_type_shorthand"])
            if agent_idx != ego_agent_idx
        ],
        "causality effect": [
            causality_effect[ego_agent_idx]
            for scene in scenes
            for agent_idx, causality_effect in enumerate(scene["remove_agent_i_ade"])
            if agent_idx != ego_agent_idx
        ],
    })

    facet_grid = sns.displot(
        data=causality_effect_df,
        x="causality effect",
        hue="agent type",
        palette="deep",
        binwidth=0.01,
        binrange=(-0.01 + 1e-8, max(causality_effect_df["causality effect"])),
        # bins=40,
        # log_scale=(False, True),
    )
    for ax in facet_grid.axes.flat:
        ax.axvline(x=0.0, color='r', linestyle=':')

    facet_grid.set_ylabels("count")
    facet_grid.fig.subplots_adjust(top=0.9)
    facet_grid.fig.suptitle(plot_title)

    plt.tight_layout()
    plt.savefig(os.path.join(logs_path, f"{plot_id} -- {plot_title}.png"))
    plt.savefig(os.path.join(logs_path, f"PDF {plot_id} -- {plot_title}.pdf"))
    plt.close()

    for agent_type in causality_effect_df["agent type"].unique():
        causality_effects = causality_effect_df[causality_effect_df["agent type"] == agent_type]["causality effect"]
        print(f"{agent_type} count: {len(causality_effects)}")
        print(f"   ==  0 cm {sum([c == 0.0 for c in causality_effects])}")
        print(f"   <   1 mm {sum([c < 0.001 for c in causality_effects])}")
        print(f"   <   1 cm {sum([c < 0.01 for c in causality_effects])}")
        print(f"   <   5 cm {sum([c < 0.05 for c in causality_effects])}")
        print(f"   <  10 cm {sum([c < 0.10 for c in causality_effects])}")
        print(f"   <   1  m {sum([c < 1.0 for c in causality_effects])}")
        print(f"   >=  0 cm {sum([c >= 0.0 for c in causality_effects])}")
        print(f"")


def plot_trajectory_curvature(scenes, logs_path, curvature_cutoff=None, plot_id="curvature",
                              plot_title="Trajectory curvature histogram", usetex=True):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_style("ticks")
    sns.set_palette("flare")
    # sns.set_palette("deep")

    plt.rc('font', **{'size': 12})
    plt.rc('figure', dpi=200)
    if usetex:
        setup_latex_for_matplotlib()

    curvature_df = pd.DataFrame({
        "agent type": [agent_type for scene in scenes for agent_type in scene["agent_type"]],
        "curvature": [curvature for scene in scenes for curvature in scene["trajectory_curvature_max"]],
    })

    facet_grid = sns.displot(
        data=curvature_df[curvature_df.curvature >= 0],
        x="curvature",
        hue="agent type",
        palette="deep",
        binwidth=0.02,
        binrange=(-0.02 + 1e-8, max(curvature_df["curvature"])),
    )
    if curvature_cutoff is not None:
        for ax in facet_grid.axes.flat:
            ax.axvline(x=curvature_cutoff, color='r', linestyle=':')

    facet_grid.set_ylabels("count")
    facet_grid.fig.subplots_adjust(top=0.9)
    facet_grid.fig.suptitle(plot_title)

    print(f"Out of {len(curvature_df)} agents, "
          f"{len(curvature_df[curvature_df.curvature > 0.5])} "
          f"had a trajectory curvature higher than 0.5.")

    plt.tight_layout()
    plt.savefig(os.path.join(logs_path, f"{plot_id} -- {plot_title}.png"))
    plt.savefig(os.path.join(logs_path, f"PDF {plot_id} -- {plot_title}.pdf"))
    plt.close()


def plot_agent_type_distribution(scenes, logs_path, plot_id="agent_type", plot_title="Agent type distribution",
                                 usetex=True):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_style("ticks")
    # sns.set_palette("flare")
    sns.set_palette("deep")

    plt.rc('font', **{'size': 12})
    plt.rc('figure', dpi=200)
    if usetex:
        setup_latex_for_matplotlib()

    agent_type_df = pd.DataFrame({
        # TODO
        # "agent type": [agent_type for scene in scenes for agent_type in scene["agent_type_shorthand"]],
        "agent type": [agent_type for scene in scenes for agent_type in
                       compute_agent_types(scene, return_shorthand=True)],
    })

    ax = sns.countplot(data=agent_type_df, x="agent type")
    for i in ax.containers:
        ax.bar_label(i, )

    plt.title(plot_title)
    ax.set_ylabel("count")

    plt.tight_layout()
    plt.savefig(os.path.join(logs_path, f"{plot_id} -- {plot_title}.png"))
    plt.savefig(os.path.join(logs_path, f"PDF {plot_id} -- {plot_title}.pdf"))
    plt.close()


def plot_agent_type_ratio_in_scene_distribution(scenes, logs_path, plot_id="agent_type_ratio",
                                                plot_title="Distribution of agent type per-scene ratio", usetex=True):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_style("ticks")
    # sns.set_palette("flare")
    sns.set_palette("deep")

    plt.rc('font', **{'size': 12})
    plt.rc('figure', dpi=200)
    if usetex:
        setup_latex_for_matplotlib()

    agent_type_df = pd.DataFrame({
        "scene idx": [scene_idx for scene_idx in range(len(scenes))],
        "agent type": [compute_agent_types(scene, return_shorthand=True) for scene in scenes],
    })
    agent_type_df = agent_type_df.explode("agent type")
    agent_type_ratios_df = agent_type_df.groupby("scene idx").value_counts(normalize=True).unstack(fill_value=0)
    agent_type_ratios_df = agent_type_ratios_df.stack().reset_index().rename(columns={0: "ratio of agent type"})

    facet_grid = sns.displot(
        data=agent_type_ratios_df,
        x="ratio of agent type",
        hue="agent type",
        palette="deep",
        binwidth=0.01,
        binrange=(-0.01 + 1e-8, 1.01 + 1e-8),
        stat="percent",
        multiple="stack",
    )

    facet_grid.set_ylabels("percentage of scenes")
    facet_grid.fig.subplots_adjust(top=0.9)
    facet_grid.fig.suptitle(plot_title)

    plt.tight_layout()
    plt.savefig(os.path.join(logs_path, f"{plot_id} -- {plot_title}.png"))
    plt.savefig(os.path.join(logs_path, f"PDF {plot_id} -- {plot_title}.pdf"))
    plt.close()


def plot_summary_figures(scenes, config, logs_path, usetex):
    plot_agent_type_distribution(scenes, logs_path, usetex=usetex)
    plot_causality_effect_per_agent_type(scenes, logs_path, usetex=usetex)
    plot_trajectory_curvature(scenes, logs_path, config.curvature_cutoff, usetex=usetex)
    plot_agent_type_ratio_in_scene_distribution(scenes, logs_path, usetex=usetex)


def plot_per_scene_figures(i, scene, config, logs_path, usetex):
    # 1. Plot png of trajectory
    plot_trajectory(
        scene=scene["trajectories"],
        figname=f'{logs_path}/factual-scene/scene-idx-{i}.png',
        lim=config.scale + .5
    )

    # 2. Plot trajectory animation
    animate(
        scene=scene["trajectories"],
        causality=scene["causality_labels"],
        figname=f'{logs_path}/factual-scene/scene-idx-{i}.gif',
        lim=config.scale + .5
    )

    # 3. Plot trajectories when individual agents get removed (i.e., the counterfactual scenarios)
    agent_removal_plots_path = os.path.join(logs_path, "counterfactual-scene")
    ensure_dir(agent_removal_plots_path)
    plot_with_agent_removal(scene, agent_removal_plots_path, plot_id=f"{i}", usetex=usetex)

    # 4. Plot png and animation of trajectory when individual agents get removed
    assert len(scene["remove_agent_i_trajectories"]) == len(scene["remove_agent_i_causality_labels"])
    for j, (trajectory_j, causality_labels_j) in enumerate(zip(
            scene["remove_agent_i_trajectories"],
            scene["remove_agent_i_causality_labels"]
    )):
        indices_of_present_agents = [agent_idx for agent_idx in range(len(trajectory_j)) if agent_idx != j]
        trajectory_j = trajectory_j[indices_of_present_agents]
        causality_labels_j = causality_labels_j[indices_of_present_agents]
        animate(
            scene=trajectory_j,
            causality=causality_labels_j,
            figname=f'{agent_removal_plots_path}/{i}_remove-A{j + 1:02d}.gif',
            lim=config.scale + .5
        )
