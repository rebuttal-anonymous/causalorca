# load data and transfer to the format that STGAT uses
import random

import numpy as np
import pytorch_lightning as pl

from causalorca.utils.util import ensure_dir

'''
data format:
data: (scences, agents, time_steps, position)
causal_data: (scences, agents, time_steps, causal relation)
output_subsets_path: str, path to the output directory (directory and parents will be created if necessary)
'''

data_path = 'data/orca_circle_crossing_2_6_ped_10000_scenes_0.3_radius_2.0_horizon.npz'
causal_data_path = 'data_causal/orca_circle_crossing_2_causality_6_ped_10000_scenes_0.3_radius_2.0_horizon.npz'
output_subsets_path = 'data/stgat/v0'

ensure_dir(output_subsets_path)


def preprocess_data(
        trajectory_data: np.array,
        causality_labels_data: np.array,
        scene_path_template: str,
        remove_causal: bool,
        remove_non_causal: bool,
        ego_agent_idx: int = 0,
        max_remove=None,
        percent_remove=None,
        seed=72,
):
    """
    Preprocess the data into a csv format digestible by STGAT.
    The created csv file is tab separated and does not have a header.
    The rows in the csv file are in the format: frame,ped_id,pos_x,pos_y.
    Flags for removing causal and non-causal agents are supported,
    where causality is defined by being or not being causal to the movement of an ego agent.

    @param trajectory_data: 4d numpy array of agent positions
        of shape (num_scenes,num_agents,time_steps,xy_coordinate).
        For example, shape could be (10,6,20,2).
    @param causality_labels_data: 4d numpy array of causality labels
        of shape (num_scenes,num_first_agents,time_steps,num_second_agent),
        where each label tells if the second agent was causal for the movement of the first agent, at a given timestep.
        For example, shape could be (10,6,20,6).
    @param scene_path_template: A template string like "scene_with_index_{i}.csv".
        Must contain a "{i}" that will be filled with the scene_idx, i.e. `scene_path_template.format(i=scene_idx)`.
    @param remove_causal: Whether to remove causal agents to the ego agent.
    @param remove_non_causal: Whether to remove non-causal agents to the ego agent.
    @param ego_agent_idx: The index of the ego agent among all scenes.
        Used only if remove_causal or remove_non_causal flags are set,
        in which case the agents that are causal or non-causal to the ego agent are removed.
    @param max_remove: Maximum number of agents to be remove per scene.
        The removed agents will be selected randomly from removable agents.
        A removable agent is the one that would be removed given the remove_causal and remove_non_causal criteria.
    @param percent_remove: Percentage of removable agents that will be removed.
        A removable agent is the one that would be removed given the remove_causal and remove_non_causal criteria.
        Bernoulli sampling is used with the given parameter. For example,
        0.33 means that for each removable agent a biased 33% heads coin flip is made,
        and if heads lands, then the agent is removed.
    """
    pl.seed_everything(seed, workers=True)

    num_scenes = trajectory_data.shape[0]
    num_agents = trajectory_data.shape[1]
    num_frames = trajectory_data.shape[2]
    num_coordinates = trajectory_data.shape[3]
    assert num_coordinates == 2

    assert causality_labels_data.shape[0] == num_scenes
    assert causality_labels_data.shape[1] == num_agents
    assert causality_labels_data.shape[2] == num_frames
    assert causality_labels_data.shape[3] == num_agents

    removed_agents_count = 0
    causal_agents_count = 0
    non_causal_agents_count = 0
    zero_mask_sum = 0
    for scene_idx in range(num_scenes):
        agent_indices_without_ego = np.array([idx for idx in range(num_agents) if idx != ego_agent_idx])

        scene_causality_labels = causality_labels_data[scene_idx, ego_agent_idx, :, :]
        causal_agents_mask = scene_causality_labels[:, agent_indices_without_ego].any(axis=0)
        non_causal_agents_mask = ~causal_agents_mask

        # mask with True if agent should be kept
        mask = np.ones_like(agent_indices_without_ego, dtype=bool)
        if remove_causal:
            mask *= ~causal_agents_mask
        if remove_non_causal:
            mask *= ~non_causal_agents_mask
        if percent_remove is not None:
            mask = mask + (~mask) * (~np.random.binomial(n=1, p=percent_remove, size=mask.shape).astype(bool))
        if max_remove is not None and (~mask).sum() > max_remove:
            removal_indices = list(zip(*np.where(mask == 0)))
            removal_indices = random.sample(population=removal_indices, k=max_remove)
            mask = np.ones_like(mask, dtype=bool)  # Reset to all ones
            for removal_idx in removal_indices:
                mask[removal_idx] = 0
            assert (~mask).sum() == max_remove
        if len(mask) > 0 and mask.sum() == 0:
            zero_mask_sum += 1

        considered_agents = [ego_agent_idx] + agent_indices_without_ego[mask].tolist()
        considered_agents = sorted(considered_agents)
        considered_agents = np.array(considered_agents)

        causal_agents_count += causal_agents_mask.sum()
        non_causal_agents_count += non_causal_agents_mask.sum()
        removed_agents_count += num_agents - len(considered_agents)

        scene_data = np.zeros((int(len(considered_agents) * num_frames), 4))
        for frame in range(num_frames):
            frame_array = np.ones((len(considered_agents), 1)) * frame * 10.0
            position_array = trajectory_data[scene_idx, :, frame, :][considered_agents, :]
            data_array = np.concatenate((frame_array, considered_agents.reshape((-1, 1)), position_array), axis=1)

            scene_data[frame * len(considered_agents): (frame + 1) * len(considered_agents), ] = data_array

        # write scene to a `.txt` file
        np.savetxt(scene_path_template.format(i=scene_idx), scene_data, fmt='%.18f', delimiter='\t')

    total_agents = num_scenes * num_agents
    print(f"-" * 72)
    print(f"PREFIX={scene_path_template}")
    print(f"-" * 72)
    if zero_mask_sum > 0:
        print(
            f"[WARNING]: All agents were removed (except for the ego agent)"
            f" in {zero_mask_sum} out of {num_scenes} scenes (i.e., {zero_mask_sum / num_scenes * 100:.2f})")
    print(f"   Number of SCENES: {num_scenes}")
    print(f"   Total number of FRAMES: {num_scenes * num_frames}")
    print(f"   Total number of REMOVED AGENTS across scenes:"
          f" {removed_agents_count}"
          f" ({removed_agents_count / total_agents * 100:.2f}%)")
    print(f"   Total number of NON REMOVED AGENTS across scenes:"
          f" {total_agents - removed_agents_count}"
          f" ({(total_agents - removed_agents_count) / total_agents * 100:.2f}%)")
    print(f"   Total number of AGENTS across scenes:"
          f" {total_agents}"
          f" ({num_scenes} ego agents + {num_scenes * (num_agents - 1)} other agents)")
    print(f"   Total number of CAUSAL AGENTS across scenes:"
          f" {causal_agents_count}"
          f" ({causal_agents_count / total_agents * 100:.2f}%)")
    print(f"   Total number of NON CAUSAL AGENTS across scenes:"
          f" {non_causal_agents_count}"
          f" ({non_causal_agents_count / total_agents * 100:.2f}%)")
    print(f"-" * 72)
    print()


traj_data = np.load(data_path)['raw']
causal_data = np.load(causal_data_path)['raw']

total = traj_data.shape[0]

# split data
train_traj = traj_data[0: int(total * 0.6)]
train_causal = causal_data[0: int(total * 0.6)]
val_traj = traj_data[int(total * 0.6): int(total * 0.8)]
val_causal = causal_data[int(total * 0.6): int(total * 0.8)]
test_traj = traj_data[int(total * 0.8): total]
test_causal = causal_data[int(total * 0.8): total]

for split, trajectories, causality_labels in [
    ("train", train_traj, train_causal),
    ("val", val_traj, val_causal),
    ("test", test_traj, test_causal),
]:
    standard_dataset_path = f"{output_subsets_path}/clean/{split}"
    remove_causal_dataset_path = f"{output_subsets_path}/remove_causal/{split}"
    remove_max_one_causal_dataset_path = f"{output_subsets_path}/remove_max_one_causal/{split}"
    remove_33percent_causal_dataset_path = f"{output_subsets_path}/remove_33percent_causal/{split}"
    remove_non_causal_dataset_path = f"{output_subsets_path}/remove_non_causal/{split}"
    remove_all_path = f"{output_subsets_path}/remove_all/{split}"

    ensure_dir(standard_dataset_path)
    ensure_dir(remove_causal_dataset_path)
    ensure_dir(remove_max_one_causal_dataset_path)
    ensure_dir(remove_33percent_causal_dataset_path)
    ensure_dir(remove_non_causal_dataset_path)
    ensure_dir(remove_all_path)

    preprocess_data(trajectories, causality_labels, f'{standard_dataset_path}/scene_{{i}}.csv', False, False)
    preprocess_data(trajectories, causality_labels, f'{remove_causal_dataset_path}/scene_{{i}}.csv', True, False)
    preprocess_data(trajectories, causality_labels, f'{remove_max_one_causal_dataset_path}/scene_{{i}}.csv', True,
                    False,
                    max_remove=1)
    preprocess_data(trajectories, causality_labels, f'{remove_33percent_causal_dataset_path}/scene_{{i}}.csv', True,
                    False, percent_remove=1 / 3)
    preprocess_data(trajectories, causality_labels, f'{remove_non_causal_dataset_path}/scene_{{i}}.csv', False, True)
    preprocess_data(trajectories, causality_labels, f'{remove_all_path}/scene_{{i}}.csv', True, True)
    print()
    print()

"""md
Example of running the script (on data previously created with `python generate_scenes.py --num_scenes 10000`).

Command: `python -m tools.stgat_preprocess`
Output:
```js
Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/clean/train/scene_{i}.csv
------------------------------------------------------------------------
   Number of SCENES: 6000
   Total number of FRAMES: 120000
   Total number of REMOVED AGENTS across scenes: 0 (0.00%)
   Total number of NON REMOVED AGENTS across scenes: 36000 (100.00%)
   Total number of AGENTS across scenes: 36000 (6000 ego agents + 30000 other agents)
   Total number of CAUSAL AGENTS across scenes: 22273 (61.87%)
   Total number of NON CAUSAL AGENTS across scenes: 7727 (21.46%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_causal/train/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 1696 out of 6000 scenes (i.e., 28.27)
   Number of SCENES: 6000
   Total number of FRAMES: 120000
   Total number of REMOVED AGENTS across scenes: 22273 (61.87%)
   Total number of NON REMOVED AGENTS across scenes: 13727 (38.13%)
   Total number of AGENTS across scenes: 36000 (6000 ego agents + 30000 other agents)
   Total number of CAUSAL AGENTS across scenes: 22273 (61.87%)
   Total number of NON CAUSAL AGENTS across scenes: 7727 (21.46%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_max_one_causal/train/scene_{i}.csv
------------------------------------------------------------------------
   Number of SCENES: 6000
   Total number of FRAMES: 120000
   Total number of REMOVED AGENTS across scenes: 5960 (16.56%)
   Total number of NON REMOVED AGENTS across scenes: 30040 (83.44%)
   Total number of AGENTS across scenes: 36000 (6000 ego agents + 30000 other agents)
   Total number of CAUSAL AGENTS across scenes: 22273 (61.87%)
   Total number of NON CAUSAL AGENTS across scenes: 7727 (21.46%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_33percent_causal/train/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 5 out of 6000 scenes (i.e., 0.08)
   Number of SCENES: 6000
   Total number of FRAMES: 120000
   Total number of REMOVED AGENTS across scenes: 7648 (21.24%)
   Total number of NON REMOVED AGENTS across scenes: 28352 (78.76%)
   Total number of AGENTS across scenes: 36000 (6000 ego agents + 30000 other agents)
   Total number of CAUSAL AGENTS across scenes: 22273 (61.87%)
   Total number of NON CAUSAL AGENTS across scenes: 7727 (21.46%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_non_causal/train/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 40 out of 6000 scenes (i.e., 0.67)
   Number of SCENES: 6000
   Total number of FRAMES: 120000
   Total number of REMOVED AGENTS across scenes: 7727 (21.46%)
   Total number of NON REMOVED AGENTS across scenes: 28273 (78.54%)
   Total number of AGENTS across scenes: 36000 (6000 ego agents + 30000 other agents)
   Total number of CAUSAL AGENTS across scenes: 22273 (61.87%)
   Total number of NON CAUSAL AGENTS across scenes: 7727 (21.46%)
------------------------------------------------------------------------



Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/clean/val/scene_{i}.csv
------------------------------------------------------------------------
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 0 (0.00%)
   Total number of NON REMOVED AGENTS across scenes: 12000 (100.00%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7525 (62.71%)
   Total number of NON CAUSAL AGENTS across scenes: 2475 (20.62%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_causal/val/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 595 out of 2000 scenes (i.e., 29.75)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 7525 (62.71%)
   Total number of NON REMOVED AGENTS across scenes: 4475 (37.29%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7525 (62.71%)
   Total number of NON CAUSAL AGENTS across scenes: 2475 (20.62%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_max_one_causal/val/scene_{i}.csv
------------------------------------------------------------------------
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 1990 (16.58%)
   Total number of NON REMOVED AGENTS across scenes: 10010 (83.42%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7525 (62.71%)
   Total number of NON CAUSAL AGENTS across scenes: 2475 (20.62%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_33percent_causal/val/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 4 out of 2000 scenes (i.e., 0.20)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 2521 (21.01%)
   Total number of NON REMOVED AGENTS across scenes: 9479 (78.99%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7525 (62.71%)
   Total number of NON CAUSAL AGENTS across scenes: 2475 (20.62%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_non_causal/val/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 10 out of 2000 scenes (i.e., 0.50)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 2475 (20.62%)
   Total number of NON REMOVED AGENTS across scenes: 9525 (79.38%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7525 (62.71%)
   Total number of NON CAUSAL AGENTS across scenes: 2475 (20.62%)
------------------------------------------------------------------------



Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/clean/test/scene_{i}.csv
------------------------------------------------------------------------
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 0 (0.00%)
   Total number of NON REMOVED AGENTS across scenes: 12000 (100.00%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7502 (62.52%)
   Total number of NON CAUSAL AGENTS across scenes: 2498 (20.82%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_causal/test/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 589 out of 2000 scenes (i.e., 29.45)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 7502 (62.52%)
   Total number of NON REMOVED AGENTS across scenes: 4498 (37.48%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7502 (62.52%)
   Total number of NON CAUSAL AGENTS across scenes: 2498 (20.82%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_max_one_causal/test/scene_{i}.csv
------------------------------------------------------------------------
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 1989 (16.57%)
   Total number of NON REMOVED AGENTS across scenes: 10011 (83.43%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7502 (62.52%)
   Total number of NON CAUSAL AGENTS across scenes: 2498 (20.82%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_33percent_causal/test/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 1 out of 2000 scenes (i.e., 0.05)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 2503 (20.86%)
   Total number of NON REMOVED AGENTS across scenes: 9497 (79.14%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7502 (62.52%)
   Total number of NON CAUSAL AGENTS across scenes: 2498 (20.82%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_non_causal/test/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 11 out of 2000 scenes (i.e., 0.55)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 2498 (20.82%)
   Total number of NON REMOVED AGENTS across scenes: 9502 (79.18%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7502 (62.52%)
   Total number of NON CAUSAL AGENTS across scenes: 2498 (20.82%)
------------------------------------------------------------------------



Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_all/train/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 6000 out of 6000 scenes (i.e., 100.00)
   Number of SCENES: 6000
   Total number of FRAMES: 120000
   Total number of REMOVED AGENTS across scenes: 30000 (83.33%)
   Total number of NON REMOVED AGENTS across scenes: 6000 (16.67%)
   Total number of AGENTS across scenes: 36000 (6000 ego agents + 30000 other agents)
   Total number of CAUSAL AGENTS across scenes: 22273 (61.87%)
   Total number of NON CAUSAL AGENTS across scenes: 7727 (21.46%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_all/val/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 2000 out of 2000 scenes (i.e., 100.00)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 10000 (83.33%)
   Total number of NON REMOVED AGENTS across scenes: 2000 (16.67%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7525 (62.71%)
   Total number of NON CAUSAL AGENTS across scenes: 2475 (20.62%)
------------------------------------------------------------------------

Global seed set to 72
------------------------------------------------------------------------
PREFIX=data/stgat/v0/remove_all/test/scene_{i}.csv
------------------------------------------------------------------------
[WARNING]: All agents were removed (except for the ego agent) in 2000 out of 2000 scenes (i.e., 100.00)
   Number of SCENES: 2000
   Total number of FRAMES: 40000
   Total number of REMOVED AGENTS across scenes: 10000 (83.33%)
   Total number of NON REMOVED AGENTS across scenes: 2000 (16.67%)
   Total number of AGENTS across scenes: 12000 (2000 ego agents + 10000 other agents)
   Total number of CAUSAL AGENTS across scenes: 7502 (62.52%)
   Total number of NON CAUSAL AGENTS across scenes: 2498 (20.82%)
------------------------------------------------------------------------




Process finished with exit code 0

```
"""
