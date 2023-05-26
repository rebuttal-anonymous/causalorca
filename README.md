# CausalOrca: An ORCA-based Diagnostic Dataset for Causally-aware Multi-agent Trajectory Prediction

This repository contains the codebase and documentation for CausalOrca, an ORCA-based synthetic trajectory prediction dataset with causality labels. The dataset is generated using a modified version of the RVO2 simulator and aims to provide annotations of ground-truth causal effects and fine-grained agent categories for social interactions in multi-agent scenarios. The documentation is a crucial component of this repository, providing comprehensive information and guidance on working with the dataset, its structure, and the underlying methodology.

To work with the dataset in your own codebase, you do not necessarily need this codebase. Simply having pickle and numpy libraries is sufficient for loading the dataset. You can directly proceed to the [Download](#download-causalorca-v1a) section.

However, there are specific cases where you may find the code in the repository useful:
1. Utilizing the dataset visualization tool provided in the [causalorca.utils.plots](src/causalorca/utils/plots.py) module. This tool can be used as-is or modified to suit your own codebase.
2. Generating a modified version of the dataset according to your specific requirements.

Feel free to explore the codebase for these purposes, while keeping in mind that it is not essential for basic usage of the dataset in your own codebase.

## Download CausalOrca v1.a

The generated synthetic data is available on Zenodo, see [here](https://doi.org/10.5281/zenodo.7973395). You can download the train, validation, and test splits using the following commands:

```bash
cd /path/to/your/data/folder
wget https://zenodo.org/record/7973395/files/synth_v1.a.filtered.train.pkl        #  13G  md5:8cc6dc33618647fbd92613e1509d8925
wget https://zenodo.org/record/7973395/files/synth_v1.a.filtered.val.pkl          # 4.3G  md5:c0a617ecb0e4f128fe90e232fdc43d95
wget https://zenodo.org/record/7973395/files/synth_v1.a.filtered.test.pkl         # 4.3G  md5:ca264f31a790e538005ca4ac929d3a77

# Check the md5sums, e.g.:
md5sum synth_v1.a.filtered.train.pkl synth_v1.a.filtered.val.pkl synth_v1.a.filtered.test.pkl
# 8cc6dc33618647fbd92613e1509d8925  synth_v1.a.filtered.train.pkl
# c0a617ecb0e4f128fe90e232fdc43d95  synth_v1.a.filtered.val.pkl
# ca264f31a790e538005ca4ac929d3a77  synth_v1.a.filtered.test.pkl
```

You can also download tiny versions of the subsets that contain only 10 scenes, which can be useful for debugging purposes. You can do so using the following commands:

```bash
wget https://zenodo.org/record/7973395/files/synth_v1.a.filtered.train.tiny.pkl   # 1.7M  md5:f78fcffe0eae22b90adb5aeebcea98a3
wget https://zenodo.org/record/7973395/files/synth_v1.a.filtered.val.tiny.pkl     # 1.7M  md5:5255c24c8448ad84d20c6ae64f0eb502
wget https://zenodo.org/record/7973395/files/synth_v1.a.filtered.test.tiny.pkl    # 1.7M  md5:6e11ee9c1451785bfef51d47db4371ad
```

To download the OOD test subsets, use the following commands:

```bash
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.1.test.300.pkl        # 424M  md5:8c372b91bf7d5b08d82f2ad8dbb712ea
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.2.1.test.300.pkl      # 220M  md5:fce9808a74f6e061a1dd4687b8a06547
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.2.2.test.300.pkl      # 120M  md5:999ea30af386213a9b26e9db545312f6
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.3.test.300.pkl        # 1.1G  md5:f28a77492e5451bcfa471d789072d23e
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.4.1.test.300.pkl      # 1.3G  md5:97cfb82741468fcdf277a588d64eb07f
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.4.2.test.300.pkl      # 5.9G  md5:e8b8bef7bc56b1c8a8eb2b1e4b5481e1
wget https://zenodo.org/record/7973395/files/synth_v1.a.odd.4.3.test.300.pkl      # 301M  md5:b13d3af7c7402c283f39cdd744d2ee72
```

Find more about how to use CausalOrca in the [docs](docs/CausalOrca.md).

## Dataset Statistics

The dataset contains a total of 125,095 scenes and 1,044,070 agents. It is divided into three subsets: training, validation, and test. The training subset consists of 75,057 scenes (60% of the dataset), while the validation and test subsets each contain 25,019 scenes (20% each). Each scene spans 20 timesteps.

The number of agents per scene varies from 6 to 12, with an average of 8.35 agents per scene. Specifically, the distribution of agents per scene is as follows: 10.9% of scenes have 6 agents, 11.6% have 7 agents, 12.8% have 8 agents, 14.3% have 9 agents, 15.6% have 10 agents, 16.7% have 11 agents, and 18.1% have 12 agents.

Considering a lower social causality threshold of 0.02 and an upper threshold of 0.1, among the 1,044,070 agents, 56.0% are direct causal to the ego agent, 3.9% are indirect causal, 24.5% are non-causal, and 15.5% do not fall into any of these categories.

The average minimum distance to the ego agent is 1.852. Furthermore, the average minimum distance to the ego agent is different for each agent type: direct-causal agents have an average minimum distance of 1.2, indirect-causal agents have an average minimum distance of 2.4, and non-causal agents have an average minimum distance of 3.2.

The figures below depict the distribution of the causal effect and minimum distance from the ego agent for various types of agents: all agents, direct causal agents, indirect causal agents, and non-causal agents. It is important to note that the presence of indirect causal agents is mostly observed when the minimum distance exceeds 5 units. In our simulator, agents closer to the ego agent are likely classified as direct causal agents since the dataset scenes are generated using a neighbor distance of 5 (refer to [causalorca.bin.generate_scenes](https://github.com/rebuttal-anonymous/causalorca/blob/main/src/causalorca/bin/generate_scenes.py#L80-L83)). Indirect causal agents can be present below a minimum distance of 5 only if they consistently remain behind the ego agent, ensuring they remain outside the ego agent's field of view at all times.

|                             Linear scale                             |                                   Log-Y scale                                    |
|:--------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| ![causal_effect_histogram.png](assets%2Fcausal_effect_histogram.png) | ![causal_effect_histogram__logy.png](assets%2Fcausal_effect_histogram__logy.png) |

|                                   Linear scale                                   |                                         Log-Y scale                                          |
|:--------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| ![min_distance_to_ego_histogram.png](assets%2Fmin_distance_to_ego_histogram.png) | ![min_distance_to_ego_histogram__logy.png](assets%2Fmin_distance_to_ego_histogram__logy.png) |

|                                    All agents                                    |                                                       Direct causal agents                                                       |                                                        Indirect causal agents                                                        |                                                    Non-causal agents                                                     |
|:--------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
| ![min_distance_to_causal_effect.png](assets%2Fmin_distance_to_causal_effect.png) | ![min_distance_to_ego_wrt_causal_effect__direct causal.png](assets%2Fmin_distance_to_ego_wrt_causal_effect__direct%20causal.png) | ![min_distance_to_ego_wrt_causal_effect__indirect causal.png](assets%2Fmin_distance_to_ego_wrt_causal_effect__indirect%20causal.png) | ![min_distance_to_ego_wrt_causal_effect__non-causal.png](assets%2Fmin_distance_to_ego_wrt_causal_effect__non-causal.png) |

For more details about the computation of the causal effect statistics and corresponding plots, please refer to [tools.compute_dataset_statistics](tools/compute_dataset_statistics.py).

## Enhanced Simulation with Visibility Constraints

In our data generation process, we utilize a customized version of the Reciprocal Velocity Obstacle 2 (RVO2) simulator. RVO2 employs Optimal Reciprocal Collision Avoidance (ORCA), upon which we introduce enhancements.

Firstly, we impose a visibility constraint in which agents recognize other agents within their close proximity or their 210&deg; field of view, facilitating the modeling of a richer spectrum of direct and indirect inter-agent influences. These enhancements are crucial since the original ORCA formulation assumes a circular attention radius, where every agent has symmetrical visibility to all others within a predefined and wide radius.

Additionally, we introduce a visibility window that maintains a record of previously visible agents. This feature allows for tracking and considering agents that were within the visibility range but have moved out of sight, enabling smoother and more realistic simulation scenarios.

## Dataset Limitations

We must acknowledge several limitations of our dataset generation approach.

Firstly, the resultant trajectories are subsampled from the underlying simulator trajectories. Careful attention is given to ensuring that the visibility window is wider than the subsampling interval to produce accurate per-timestep direct causality labels. However, the causal effect we compute cannot take into account possible influences at unsampled points.

Secondly, our per-timestep direct causality labels are based on whether an agent was used in the ORCA formula to calculate the next position of an agent. This formulation does not measure a causal effect, unlike the causality labels based on counterfactual scenarios that we employ. It might therefore be prone to reporting false positives, such as a non-causal agent having a direct impact on another agent at a specific timestep.

Lastly, the imposition of visibility constraints in our data generation process deviates from the original ORCA formulation. The original formulation assumes symmetric visibility, enabling the use of convex optimization techniques for efficient goal attainment and collision avoidance. However, by introducing visibility constraints, we acknowledge that agents may not have symmetric visibility, leading to differences in their optimization processes.

This discrepancy in visibility can sometimes result in unexpected scenarios, where an agent is surprised to realize that another agent did not perceive its presence. Consequently, the non-perceiving agent did not optimize the same convex optimization problem, leading to potentially unnatural and jerky trajectories. To mitigate this issue, we introduce a visibility window that maintains a record of previously visible agents. This window helps alleviate the effects of sudden visibility changes and allows for smoother trajectories.

## Environment set-up

This codebase has been tested with the packages and versions specified in `requirements.txt` and Python 3.9 on Manjaro Linux and Red Hat Enterprise Linux Server 7.7 (Maipo).

Start by cloning the repository:

```bash
git clone https://github.com/rebuttal-anonymous/causalorca.git
```

With the repository cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:

```bash
conda create -n causalorca python=3.9 -y
conda activate causalorca
```

Then, install [PyTorch](https://pytorch.org/) 1.12.1 and [torchvision](https://pytorch.org/vision/stable/index.html) 0.13.1. For example with CUDA 11 support:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

```

Install our fork of [Python-RVO2](https://github.com/rebuttal-anonymous/Python-RVO2) into your Python environment. Python-RVO2 contains Cython-based Python bindings for the RVO2 framework that we use for multi-agent simulation. To install it, follow either the official instructions or the following outline:

```bash
export PYTHON_RVO2_INSTALL_PATH=/some/path/where/you/will/clone

git clone https://github.com/rebuttal-anonymous/Python-RVO2.git $PYTHON_RVO2_INSTALL_PATH
cd $PYTHON_RVO2_INSTALL_PATH

pip install Cython
python setup.py build
python setup.py install

cd -
```

If Python-RVO2 was installed correctly, importing the module in Python should not give any errors: `python -c "import rvo2"`. If however, you want to modify Python-RVO2 itself (for example to change the C++ code that defines agents' neighborhood), then you will need to rebuild and reinstall the package to apply the changes made. This could, for example, be done like: `rm -rf build/ cmake-build-debug/ dist/ pyrvo2.egg-info/; python setup.py build; pip uninstall pyrvo2 -y; python setup.py install`.

With Python-RVO2 set up, install the remaining required packages:

```bash
pip install -r requirements.txt
```

Finally, make sure to add the source code to your Python path if running scripts. Note that you have to re-do this each time you create a shell session. For example:
```bash
export PYTHONPATH="$PYTHONPATH:`pwd`/src"
```

_TODO: There is a simpler method to install Python-RVO2, which involves running the following command: `python -m pip install git+https://github.com/rebuttal-anonymous/Python-RVO2`._

_TODO: It has come to our attention that the installation process for Python-RVO2 can be challenging on certain systems. In order to ensure that the dataset is self-contained and independent of the simulator, it is essential to make it easily accessible to the community. To achieve this, we propose a clear separation between the simulator and the tools utilized for the dataset. One approach is to categorize the requirements into separate files, such as `requirements/common.txt` and `requirements/dev.txt`, as suggested [here](https://stackoverflow.com/questions/17803829/how-to-customize-a-requirements-txt-for-multiple-environments). Alternatively, a division can be made between `requirements.txt` and `dev_requirements.txt`, as done in [robustnav](https://github.com/allenai/robustnav/blob/main/dev_requirements.txt)._
