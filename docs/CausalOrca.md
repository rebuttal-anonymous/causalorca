# CausalOrca v1.a
CausalOrca v1.a represents the initial release of a synthetic dataset generated using a modified version of the RVO2 simulator.

This document provides an overview of the dataset, covering topics such as (i) getting started with utilizing the dataset, (ii) what is inside the dataset, (iii) dataset visualization, and (iv) dataset generation.

## Getting started with the dataset

To begin using the dataset, you can refer to the [`tools.playground`](../tools/playground.py) module. Running the script and examining its functionality will provide insights into working with the dataset. You can adapt similar code to load and integrate the dataset into your own codebase. Notably, the causalorca codebase is primarily required for generating and visualizing the dataset. Loading the data in your own codebase only requires the pickle and numpy libraries.

## What is inside the dataset

The dataset comprises a collection of scenes and a configuration file used during dataset generation. These components can be loaded as follows:

```python
DATASET_PATH="data/synth_v1.a.filtered.test.tiny.pkl"
with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)
scenes = dataset["scenes"]
config = dataset["config"]
```

### The scenes

Every scene in the dataset holds the following data:
- `trajectories`: A numpy.ndarray of `numpy.float64` of shape `(n_agents,n_timesteps,2)` (e.g., `(11, 20, 2)`) containing the trajectory of each agent in the scene.
- `validity`: A boolean value indicating whether the scene is valid or invalid, with invalid scenes being non-smooth.
- `remove_agent_i_trajectories`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents_removed,n_agents,n_timesteps,2)` (e.g., `(11, 11, 20, 2)`) representing the trajectories of non-removed agents when one agent is removed. This provides counterfactual outcomes for individual agent removal. The trajectory of the removed agent is set to `numpy.nan`. For example, if agent with index 3 was removed, then `scene['remove_agent_i_trajectories'][3, :,:,:]` would contain the trajectories of all non-removed agents, with the trajectory of the removed agent `scene['remove_agent_i_trajectories'][3,3, :,:]` being set to `numpy.nan`. Encountering `numpy.nan` values means that the data structure is not used correctly.
- `remove_agent_i_validity`: A `numpy.ndarray` of `bool` of shape `(n_agents,)` (e.g., `(11,)`) indicating the validity of the counterfactual scenes when a certain agent is removed.
- `remove_agent_i_ade`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents_removed,n_agents)` saying for each agent what the Average Displacement Error (ADE) between the factual and counterfactual scenario was when an agent was removed. For example, if the agent with index 3 was removed, then `scene['remove_agent_i_ade][3,6]` would tell the ADE for the agent with index 6 when comparing the counterfactual trajectory `scene['remove_agent_i_trajectories'][3,6, :,:]` with the factual trajectory `scene['trajectories'][6, :,:]`. If the ADE is exactly equal to zero, then we can formally claim that there was no causal effect on the trajectory of another agent, given that the treatment was the removal of one agent.
- `causality_labels`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents_**to**,n_timesteps, n_agents_**from**)` (e.g., `(11, 20, 11)`) representing **per-timestep** causal influence between agents. A value of 1 indicates a direct causal link, while 0 indicates no link. Aggregating these labels might say that an agent had a direct causal link at all timesteps, but still had a `remove_agent_i_ade` of zero.
- `remove_agent_i_causality_labels`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents_removed,n_agents_**to**,n_timesteps, n_agents_**from**)` (e.g., `(11, 11, 20, 10)`) which says if there exists a direct causal link from one agent to another agent in the counterfactual scenario of removing one agent. For example, if an agent with index 3 was removed, then `scene['remove_agent_i_causality_labels'][3, :,:,:]` corresponds to the existence of direct causality links among the other agents, with `scene['remove_agent_i_causality_labels'][3,3, :,:]` having dummy values of `numpy.nan`, as the agent with index three was removed from the scene.
- `scenario_configuration`: A list of dictionaries that contain the initial scene setup configuration given to the simulator (like the initial positions of the agents and their goals). Can be used to create the original [`causalorca.orca.scenario.scenario_configuration.SceneConfiguration`](../src/causalorca/orca/scenario/scenario_configuration.py) scenario and reproduce the simulation.
- `agent_setup_type`: A list of strings indicating the setup type of each agent in the scenario. Depending on the scene layout used, the names used to describe the setup type will be different. For example, in the mixed scenario 2, the agent setup types are any of the following: "static", "circle_crossing", "square_crossing", "leader_follower", "parallel_overtake". They can also be fetched from the `scenario_configuration`, but are extracted here for convenience.
- `scene_name`: A string representing the name assigned to the scene during dataset generation. Note that it may not be unique after merging datasets, so it is not recommended to use it as a unique identifier - use the datapoint index instead.
- `trajectory_interaction`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents,)` (e.g., (11,)), indicating the estimated amount of interaction for an agent's trajectory. Consult [causalorca.orca.estimate_trajectory_curvature](../src/causalorca/orca/estimate_trajectory_curvature.py) to see how it is computed.
- `trajectory_curvature_mean`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents,)` (e.g., (11,)), saying what the estimated max-over-timestep curvature of an agent's trajectory is. Consult [causalorca.orca.estimate_trajectory_curvature](../src/causalorca/orca/estimate_trajectory_curvature.py) to see how it is computed.
- `trajectory_curvature_max`: A `numpy.ndarray` of `numpy.float64` of shape `(n_agents,)` (e.g., (11,)), saying what the estimated mean-over-timestep curvature of an agent's trajectory is. Consult [causalorca.orca.estimate_trajectory_curvature](../src/causalorca/orca/estimate_trajectory_curvature.py) to see how it is computed.

Deprecated data:
- `indirect_causality_effect` **`[DEPRECATED]`**: A `numpy.ndarray` of `bool` of shape `(n_agents_removed,n_agents)` (e.g., `(11, 11)`), telling whether there was an indirect causal effect in the causal graph defined by `causality_labels`.
- `estimated_causality_label` **`[DEPRECATED]`**: A list of strings telling the causality label of each agent wrt to the ego agent. Those labels are enumerated in [causalorca.orca.constants](../src/causalorca/orca/constants.py), alongside the functionality to compute the labels given a scene.
- `estimated_causality_label_shorthand` **`[DEPRECATED]`**: A list of strings like `estimated_causality_label`, but containing a shorthand of the label (e.g., "C" instead of "Causal (C)").
- `agent_type` **`[DEPRECATED]`**: A list of strings telling a per-agent type. Currently equals to `estimated_causality_label`, i.e., the agent type is defined as the causality label.
- `agent_type_shorthand` **`[DEPRECATED]`**: A list of strings like `agent_type`, but containing a shorthand of the type.

### The configuration file

The configuration is a dump of the parser arguments given to the dataset generation script. To interpret them, head over to [causalorca.bin.generate_scenes](../src/causalorca/bin/generate_scenes.py). For example, the following were the arguments used when generating the first 10k datapoints of the `v1.a` dataset.
```json
{
    'simulator': 'orca',
    'layout': 'mixed_scenario_2',
    'scale': 5.0,
    'min_num_ped': 6,
    'max_num_ped': 12,
    'num_scenes': 10000,
    'min_dist': 0.5,
    'seed': 1,
    'logs_path': './logs/test_synth_v1.a_1__2022-11-16_00:25:34.274991875/',
    'data_output_dir': './logs/test_synth_v1.a_1__2022-11-16_00:25:34.274991875/',
    'radius': 0.3,
    'max_speed': 1.0,
    'fov': 210.0,
    'very_close_ratio': 0.1,
    'visibility_window': 4,
    'neighbor_dist': 5.0,
    'horizon': 2.0,
    'fps': 50,
    'curvature_cutoff': 0.0,
    'max_length': 10000,
    'trajectory_length': 20,
    'observation_length': 8,
    'scene_must_have_nc_ego': False,
    'scene_must_have_nc_all': False,
    'visualize': False,
    'visualize_agent_removal': False,
    'no_latex': True,
    'log_every': 100
}
```


## Visualize CausalOrca

We strongly recommend setting up a tex environment on your system to have better-looking visuals, ready for publication. A quick installation guide for TeX Live on Linux can be found, for example, [here](https://www.tug.org/texlive/quickinstall.html). Note that the full installation might take a few hours (took 8 hours on my machine). If you do not have a tex environment and do not want to install one, be sure to pass the `--no_latex` flag to make the visualization script runnable.

To create summary visualizations of the full dataset, run the following.

```bash
python -m causalorca.bin.visualize_scenes --data_pickle_path data/synth_v1.a.full.pkl --logs_path data/visualizations/synth_v1.a.full/ --summary_only
```

To visualize all scenes in the tiny dataset, run the following (nota bene: this might take some time to finish):

```bash
python -m causalorca.bin.visualize_scenes --data_pickle_path data/synth_v1.a.tiny.pkl --logs_path data/visualizations/synth_v1.a.tiny/

```


## Generate CausalOrca

We will generate the data based on a mixed scenario layout that combines different agent configurations (circle crossing, square crossing, leader-follower agents) and contains from 6 to 12 pedestrians. Note that you need to set up the environment first, following the instructions in the [README](../README.md#environment-set-up).

The scripts below generate a dataset of 100k scenes. This is a rather large number of scenes, but we prefer the redundancy since we might have to filter the datasets with certain criteria, making the dataset smaller. For convenience, we also package the dataset into smaller portions, giving the following three subsets:
- `full`: all 100k scenes
- `mini`: only 10k scenes
- `tiny`: only  1k scenes

The scenes in the `mini` and `tiny` subsets are equivalent to the first 1k and 10k scenes in the `full` dataset, respectively. 

```bash
## 1. Generate data in 10 parallel processes, each process using another seed
for SEED in {01..50}; do
    python -m causalorca.bin.generate_scenes --layout mixed_scenario_2 --num_scenes 10000 --curvature_cutoff 0 --min_num_ped 6 --max_num_ped 12 --logs_path ./logs/test_synth_v1.a_${SEED}__`date +"%F_%T.%N"`/ --seed $SEED --no_latex &
done;

## 2. Package a tiny version of the dataset, of 1k samples
python -m tools.merge_generated_datasets --dataset_paths logs/test_synth_v1.a_01_*/*.pkl --output_merged_dataset_path "data/synth_v1.a.tiny.pkl" --scene_number_limit 1000

## 3. Package a mini version of the dataset, of 10k samples
python -m tools.merge_generated_datasets --dataset_paths logs/test_synth_v1.a_01_*/*.pkl --output_merged_dataset_path "data/synth_v1.a.mini.pkl"

## 4. Package the full dataset of 100k samples by merging the outputted data.
##    Before merging, you might want to check that your asterisks expand as you want them to expand (see the echo command below).
echo Will merge the following: logs/test_synth_v1.a_*/*.pkl
python -m tools.merge_generated_datasets --dataset_paths logs/test_synth_v1.a_*/*.pkl    --output_merged_dataset_path "data/synth_v1.a.full.pkl"
```

### Filtering the dataset

We additionally want to have a dataset of scenes filtered by the curvature of the ego agent, such that the remaining scenes have a higher among of curvature for the ego agent. To do this, we use the [`tools.filter_scenes_by_curvature`](../tools/filter_scenes_by_curvature.py) script as follows:

```bash
python -m tools.filter_scenes_by_curvature --input_data_pickle_path data/synth_v1.a.full.pkl --output_data_pickle_path data/synth_v1.a.filtered.pkl --curvature_cutoff 0.25
# Max per-timestep curvature of ego agent:
#     ==   0 cm^2      0 of 500000 (0.00%)
#     >    1 cm^2 498838 of 500000 (99.77%)
#     >    1 dm^2 463889 of 500000 (92.78%)
#     > 0.05  m^2 358213 of 500000 (71.64%)
#     > 0.10  m^2 266302 of 500000 (53.26%)
#     > 0.11  m^2 251618 of 500000 (50.32%)
#     > 0.12  m^2 238054 of 500000 (47.61%)
#     > 0.13  m^2 225545 of 500000 (45.11%)
#     > 0.14  m^2 213881 of 500000 (42.78%)
#     > 0.15  m^2 202954 of 500000 (40.59%)
#     > 0.18  m^2 174277 of 500000 (34.86%)
#     > 0.20  m^2 158082 of 500000 (31.62%)
#     > 0.25  m^2 125095 of 500000 (25.02%)
#     > 0.30  m^2 101038 of 500000 (20.21%)
#     > 0.50  m^2  47871 of 500000 (9.57%)
#
# Mean curvature of ego agent:
#     ==   0 cm^2      0 of 500000 (0.00%)
#     >    1 cm^2 498140 of 500000 (99.63%)
#     >    1 dm^2 419711 of 500000 (83.94%)
#     > 0.05  m^2 239813 of 500000 (47.96%)
#     > 0.10  m^2 136490 of 500000 (27.30%)
#     > 0.11  m^2 123235 of 500000 (24.65%)
#     > 0.12  m^2 111773 of 500000 (22.35%)
#     > 0.13  m^2 101696 of 500000 (20.34%)
#     > 0.14  m^2  92825 of 500000 (18.57%)
#     > 0.15  m^2  84826 of 500000 (16.97%)
#     > 0.18  m^2  65536 of 500000 (13.11%)
#     > 0.20  m^2  55578 of 500000 (11.12%)
#     > 0.25  m^2  37656 of 500000 (7.53%)
#     > 0.30  m^2  26625 of 500000 (5.33%)
#     > 0.50  m^2   9682 of 500000 (1.94%)
#
#
# Number of scenes BEFORE filtering: 500000
# Number of agents distribution BEFORE filtering: Counter({7: 71674, 8: 71636, 6: 71472, 9: 71408, 10: 71382, 12: 71225, 11: 71203})
# Number of scenes AFTER filtering: 125095
# Number of agents distribution AFTER filtering: Counter({12: 22693, 11: 20874, 10: 19463, 9: 17833, 8: 16073, 7: 14570, 6: 13589})
```

### Split the dataset

For convenience, we split the dataset into a train, test, and  validation subset:
```bash
python -m tools.split_scenes --input_data_pickle_path data/synth_v1.a.filtered.pkl \
 --output_train_path data/synth_v1.a.filtered.train.pkl \
 --output_val_path   data/synth_v1.a.filtered.val.pkl \
 --output_test_path  data/synth_v1.a.filtered.test.pkl \
 --val_ratio 0.2 \
 --test_ratio 0.2
 ```


## Prepare OOD datasets

The OOD datasets are generated in the following sbatch bash scripts:
```bash
bash tools/generate_ood_datasets/generate_ood_1.sh
bash tools/generate_ood_datasets/generate_ood_2.1.sh
bash tools/generate_ood_datasets/generate_ood_2.2.sh
bash tools/generate_ood_datasets/generate_ood_3.sh
bash tools/generate_ood_datasets/generate_ood_4.1.sh
bash tools/generate_ood_datasets/generate_ood_4.2.sh
bash tools/generate_ood_datasets/generate_ood_4.3.sh
```