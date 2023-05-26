"""
Script that takes a dataset as input and filters out the scenes
that do not have enough curvature for the ego agent trajectory.
The filtering is done on top of the `trajectory_curvature_max` scene attribute.
All scenes below a given threshold will be discarded.
The processed dataset is saved to the specified output directory.

Example of usage:
python -m tools.filter_scenes_by_curvature --input_data_pickle_path data/synth_v1.a.full.pkl --output_data_pickle_path data/synth_v1.a.filtered.pkl --curvature_cutoff 0.2
"""
import argparse
import os.path
import pickle
from collections import Counter

import numpy as np


def print_curvature_debug_logs(scenes):
    ego_max_curvatures = np.array([s["trajectory_curvature_max"][args.ego_agent_index] for s in scenes])
    ego_mean_curvatures = np.array([s["trajectory_curvature_mean"][args.ego_agent_index] for s in scenes])
    for curvature_name, curvatures in [
        ("Max per-timestep curvature of ego agent", ego_max_curvatures),
        ("Mean curvature of ego agent", ego_mean_curvatures)
    ]:
        print(f"{curvature_name}:")
        n = len(curvatures)

        x = (curvatures == 0).sum()
        print(f"    ==   0 cm^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.0001).sum()
        print(f"    >    1 cm^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.01).sum()
        print(f"    >    1 dm^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.05).sum()
        print(f"    > 0.05  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.1).sum()
        print(f"    > 0.10  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.11).sum()
        print(f"    > 0.11  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.12).sum()
        print(f"    > 0.12  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.13).sum()
        print(f"    > 0.13  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.14).sum()
        print(f"    > 0.14  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.15).sum()
        print(f"    > 0.15  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.18).sum()
        print(f"    > 0.18  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.20).sum()
        print(f"    > 0.20  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.25).sum()
        print(f"    > 0.25  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.30).sum()
        print(f"    > 0.30  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        x = (curvatures > 0.50).sum()
        print(f"    > 0.50  m^2 {x:>6d} of {n:>6d} ({x / n * 100:>2.2f}%)")

        print()
    print()


def main(args):
    print(f"Loading dataset from {args.input_data_pickle_path}...")
    with open(args.input_data_pickle_path, "rb") as f:
        dataset = pickle.load(f)
    print("Dataset loaded")
    print("")

    scenes = dataset["scenes"]
    print_curvature_debug_logs(scenes)

    agents_per_scene = [len(s["trajectories"]) for s in scenes]
    print(f"Number of scenes BEFORE filtering: {len(scenes):>6d}")
    print(f"Number of agents distribution BEFORE filtering: {Counter(agents_per_scene)}")

    scenes = [s for s in scenes if s["trajectory_curvature_max"][0] > args.curvature_cutoff]
    dataset["scenes"] = scenes
    dataset["__filter_scenes_by_curvature_args"] = args

    agents_per_scene = [len(s["trajectories"]) for s in scenes]
    print(f"Number of scenes AFTER filtering: {len(scenes):>6d}")
    print(f"Number of agents distribution AFTER filtering: {Counter(agents_per_scene)}")

    print(f"Saving the processed dataset to: {os.path.abspath(args.output_data_pickle_path)}")
    with open(args.output_data_pickle_path, "wb") as f:
        pickle.dump(dataset, f)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_pickle_path', type=str, required=True,
                        help='Path to pickle file of dataset to be filtered.')
    parser.add_argument('--output_data_pickle_path', type=str, required=True,
                        help='Path to pickle file to be outputted by filtering the inputted dataset.')
    parser.add_argument('--ego_agent_index', type=int, default=0,
                        help='Index of the ego agent. The filtering is done '
                             'based on the curvature value of the ego agent trajectory.')
    parser.add_argument('--curvature_cutoff', type=float, default=0.12,
                        help='The minimum allowable ego agent trajectory curvature amount. '
                             'Scenes with a lower value of ego agent trajectory curvature will '
                             'be considered as invalid and regenerated until the criteria is met.')
    args = parser.parse_args()
    print(f"args={args}")
    main(args)
