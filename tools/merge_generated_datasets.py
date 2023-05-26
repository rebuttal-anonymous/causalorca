"""
Script that takes a list of datasets (or dataset subsets) and outputs a merged dataset.
Post-processing is applied on per-scene basis that converts the scenario_configuration attribute to a dictionary.
Useful for merging datasets created with in different processes/nodes.

A few examples of how to run:

# Example 1: Just postprocess one dasaset
python -m tools.merge_generated_datasets \
    --dataset_paths "logs/test_synth_v1.a_1__2022-11-16_00:25:34.274991875/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    --output_merged_dataset_path "data/synth_v1.a.mini.npz"

# Example 2: Merge 10 datasets
python -m tools.merge_generated_datasets \
    --dataset_paths \
    "logs/test_synth_v1.a_01__2022-11-16_00:25:34.274991875/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_02__2022-11-16_00:25:34.275744004/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_03__2022-11-16_00:25:34.275143185/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_04__2022-11-16_00:25:34.275706037/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_05__2022-11-16_00:25:34.276215017/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_06__2022-11-16_00:25:34.277592688/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_07__2022-11-16_00:25:34.277194210/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_08__2022-11-16_00:25:34.277759428/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_09__2022-11-16_00:25:34.278198579/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    "logs/test_synth_v1.a_10__2022-11-16_00:25:34.278718744/synthetic_dataset__orca_layout-mixed_scenario_2_ped-in-[6,12]_scenes-10000_r-0.3_h-2.0.npz" \
    --output_merged_dataset_path \
    "data/synth_v1.a.npz"
"""
import argparse
import os.path
import pickle

from tqdm import tqdm

from causalorca.orca.scenario.scenario_configuration import SceneConfiguration
from causalorca.utils.util import ensure_dir


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def scene_post_processing(scene: dict):
    if type(scene.get("scenario_configuration", None)) == SceneConfiguration:
        assert SceneConfiguration.from_dicts(scene["scenario_configuration"].to_dicts())
        scene["scenario_configuration"] = scene["scenario_configuration"].to_dicts()
    return scene


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_paths', nargs='+', required=True,
                        help="Paths to datasets (i.e., pickle files) to be merged.")
    parser.add_argument('--output_merged_dataset_path', type=str, required=True,
                        help="Path to merged dataset to be outputted.")
    parser.add_argument('--scene_number_limit', type=int, default=None,
                        help="Limit the number of scenes in the merged dataset to the given value."
                             " No limit if left as None.")

    args = parser.parse_args()
    print("args:")
    print(args.__dict__)
    ensure_dir(os.path.dirname(args.output_merged_dataset_path))

    datasets = [
        {
            "path": dataset_path,
            "dataset": load_from_pickle(dataset_path),
        }
        for dataset_path in args.dataset_paths
    ]

    merged_dataset = {
        "scenes": [],
        "config": None,
        "__scenes_per_merged_dataset": [],
        "__all_merged_configs": [],
        "__merge_args": args.__dict__,
    }
    for dataset_dict in datasets:
        scenes = dataset_dict['dataset']['scenes']
        print(dataset_dict["path"])
        print(f"\tlen(scenes)={len(scenes)}")
        merged_dataset["scenes"] += scenes
        merged_dataset["__scenes_per_merged_dataset"] += [len(scenes)]
        merged_dataset["__all_merged_configs"] += [dataset_dict["dataset"]["config"]]
        if merged_dataset["config"] is None:
            merged_dataset["config"] = dataset_dict["dataset"]["config"]
    print()

    if args.scene_number_limit is not None:
        assert args.scene_number_limit > 0
        merged_dataset["scenes"] = merged_dataset["scenes"][:args.scene_number_limit]

    print(f"Merged dataset number of scenes: {len(merged_dataset['scenes'])}")
    for scene_idx in tqdm(range(len(merged_dataset["scenes"]))):
        merged_dataset["scenes"][scene_idx] = scene_post_processing(merged_dataset["scenes"][scene_idx])
    with open(args.output_merged_dataset_path, "wb") as f:
        pickle.dump(merged_dataset, f)
    print(f"output_merged_dataset_path:\n{os.path.abspath(args.output_merged_dataset_path)}")
