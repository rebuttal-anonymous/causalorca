"""
This is a dataset playground script that explores the CausalSynth v1.a dataset and briefly demonstrates how to work with it.

Example of how to run the script on the tiny dataset:
PYTHONPATH="$PYTHONPATH:`pwd`/src" python -m tools.playground --data_pickle_path data/synth_v1.a.tiny.pkl
"""

import argparse
import os
import pickle


def main(args):
    print(f"Loading dataset from {args.data_pickle_path}...")
    with open(args.data_pickle_path, "rb") as f:
        dataset = pickle.load(f)
    print("Dataset loaded")
    print("")

    scenes = dataset["scenes"]
    config = dataset["config"]

    print("1. What configuration was used to create the dataset?")
    print(f"\tconfig={config}")
    print()

    print("2. How many scenes are there in the dataset?")
    print(f"\tlen(scenes)={len(scenes)}")
    print()

    # TODO document what all the keys are
    print("3. What data does one scene contain?")
    print(f"\tscenes[0].keys()={scenes[0].keys()}")
    print(f"\t")
    # dict_keys(['trajectories', 'causality_labels', 'validity', 'remove_agent_i_trajectories',
    # 'remove_agent_i_causality_labels', 'remove_agent_i_validity', 'remove_agent_i_ade',
    # 'indirect_causality_effect', 'agent_type', 'estimated_causality_label', 'agent_type_shorthand',
    # 'estimated_causality_label_shorthand', 'scenario_configuration', 'agent_setup_type', 'scene_name',
    # 'trajectory_interaction', 'trajectory_curvature_mean', 'trajectory_curvature_max'])

    print("4. What is the trajectory of the ego agent in the first scene?")
    print(f"\tscenes[0]['trajectories'][0]={scenes[0]['trajectories'][0]}")
    print()

    print("5. What is the causal effect of the third agent on the ego agent, in the first scene?")
    print(f"\tscenes[0]['remove_agent_i_ade'][2][0]={scenes[0]['remove_agent_i_ade'][2][0]}")
    print()

    # TODO: Remove the deprecated labels
    print("6. [DEPRECATED AGENT CATEGORIES] What is the label of the third agent (wrt to the ego agent) "
          "in the first scene: C, NC-ego, or NC-all?")
    print(f"\tscenes[0]['estimated_causality_label'][2]={scenes[0]['estimated_causality_label'][2]}")
    print(
        f"\tscenes[0]['estimated_causality_label_shorthand'][2]={scenes[0]['estimated_causality_label_shorthand'][2]}")
    print(f"\tNota bene: These labels can also be re-computed")
    print(f"\t           for any agent as the ego agent using")
    print(f"\t           `causalorca.orca.constants.compute_estimated_causality_label`")
    print(f"\t           But note that changing the index of the ego agent")
    print(f"\t           induces a distribution shift, since the scenarios")
    print(f"\t           were built in a way that the ego agent was placed")
    print(f"\t           in particular locations/movements. For example,")
    print(f"\t           in CausalOrca v1.a, the ego agent always does")
    print(f"\t           a circle crossing, whereas other agents also do")
    print(f"\t           a square crossing and a leader-follower setup.")
    print()

    print("7. How can I plot some dataset visualizations?")
    print(f"\tIf using the causalorca codebase (i.e., this codebase), then you could use the code below. "
          f"Note that the agent categories in visualizations are deprecated.")
    from causalorca.utils.util import ensure_dir, get_str_formatted_time
    from causalorca.utils.plots import plot_summary_figures, plot_per_scene_figures

    logs_path = f"logs/playground/{get_str_formatted_time()}"
    ensure_dir(logs_path)

    # Summary visualizations for all scenes
    plot_summary_figures(
        scenes=scenes,
        config=config,
        logs_path=logs_path,
        usetex=not args.no_latex,
    )
    # Per scene visualizations
    plot_per_scene_figures(
        i=0,
        scene=scenes[0],
        config=config,
        logs_path=logs_path,
        usetex=not args.no_latex,
    )
    print(f"All visualizations generated and saved to: {os.path.abspath(logs_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pickle_path', type=str, required=True,
                        help='Path to dataset pickle file')
    parser.add_argument('--no_latex', action='store_true',
                        help='Whether not to use latex when creating plots.'
                             ' Some systems might not have a tex distribution installed.')
    args = parser.parse_args()
    print(f"args={args}")
    main(args)
