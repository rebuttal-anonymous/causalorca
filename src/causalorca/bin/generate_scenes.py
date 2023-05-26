"""
Adapted from https://github.com/vita-epfl/trajnetplusplusdataset/blob/master/trajnetdataset/controlled_data.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import Counter

import pytorch_lightning as pl
from tqdm import tqdm

from causalorca.orca.constants import compute_estimated_causality_label, estimated_causality_label
from causalorca.orca.estimate_trajectory_curvature import \
    estimate_future_trajectory_curvature_by_linearly_extrapolating_past as estimate_curvature, \
    estimate_trajectory_interaction_amount_using_current_to_future_velocity_differences as estimate_interaction
from causalorca.orca.generate_trajectory import generate_orca_trajectory
from causalorca.orca.scenario.generate_agent import generate_circle_crossing
from causalorca.orca.scenario.generate_scenario import generate_mixed_scenario_1, generate_mixed_scenario_2, \
    generate_walkway
from causalorca.utils.plots import plot_trajectory, plot_with_agent_removal, \
    plot_causality_effect_per_agent_type, plot_trajectory_curvature
from causalorca.utils.util import HORSE, nice_print, ensure_dir, get_str_formatted_time, \
    redirect_stdout_and_stderr_to_file

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np

from causalorca.utils.visualize_as_animation import animate


def parse_arguments():
    # common
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='orca',
                        choices=('orca', 'sf'))
    parser.add_argument('--layout', default='mixed_scenario_2',
                        choices=['circle_crossing',
                                 'mixed_scenario_1',
                                 'mixed_scenario_2',
                                 'mixed_scenario_3',
                                 'mixed_scenario_4',
                                 'mixed_scenario_5',
                                 'mixed_scenario_6',
                                 'mixed_scenario_7',
                                 'walkway'])
    parser.add_argument('--scale', default=5.0,
                        help='Scale of layout')
    parser.add_argument('--min_num_ped', type=int, default=7,
                        help='Minimum number of pedestrians in the scene. '
                             'Number sampled uniformly at random for each scene.')
    parser.add_argument('--max_num_ped', type=int, default=11,
                        help='Maximum number of pedestrians in the scene. '
                             'Number sampled uniformly at random for each scene.')
    parser.add_argument('--num_scenes', type=int, default=16,
                        help='Number of scenes')
    parser.add_argument('--min_dist', type=float, default=0.5,
                        help='Collision threshold')
    parser.add_argument('--seed', type=int, default=72,
                        help='Random seed')
    parser.add_argument('--logs_path', type=str, default=f"./logs/synth_logs_{get_str_formatted_time()}",
                        help='Path to save logs')
    parser.add_argument('--data_output_dir', type=str, default=None,
                        help='Path to output data to. If None, then the logs_path is used.')
    # style
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Radius of agent/robot used in ORCA simulation.')
    parser.add_argument('--max_speed', type=float, default=1.0,
                        help='Max speed of agent/robot used in ORCA simulation.')
    parser.add_argument('--fov', type=float, default=210.0,
                        help='Field-of-view of agent/robot used in ORCA simulation.')
    parser.add_argument('--very_close_ratio', type=float, default=0.1,
                        help='Very-close-ratio of agent/robot used in ORCA simulation.')
    parser.add_argument('--visibility_window', type=int, default=4,
                        help='Neighbour visibility window size of agent/robot used in ORCA simulation.')
    parser.add_argument('--neighbor_dist', type=float, default=5.0,
                        help='Neighbourhood distance in ORCA simulator. '
                             'If an agent is closer than this value, '
                             'then he is considered as a neighbour to the other agent.')
    parser.add_argument('--horizon', type=float, default=2.0,
                        help='Simulation horizon used in the ORCA simulator. '
                             'Defines the amount of time for which the simulator would guarantee no collisions.')
    parser.add_argument('--fps', type=float, default=50,
                        help='The FPS used in the ORCA simulator.')
    parser.add_argument('--curvature_cutoff', type=float, default=0.12,
                        help='The minimum allowable ego agent trajectory curvature amount. '
                             'Scenes with a lower value of ego agent trajectory curvature '
                             'will be considered as invalid and regenerated until the criteria is met.')
    # data
    parser.add_argument('--max_length', type=int, default=10000,
                        help='Max length of trajectory')
    parser.add_argument('--trajectory_length', type=int, default=20,
                        help='Exact length of trajectory to be outputted.'
                             ' E.g., 20 is a common choice.')
    parser.add_argument('--observation_length', type=int, default=8,
                        help='Number of observation timesteps.'
                             ' E.g., if the length of the trajectory is 20 timesteps,'
                             ' a common choice for the observation_length is 8 timesteps.')
    parser.add_argument('--scene_must_have_nc_ego', action='store_true',
                        help='Scene must have at least 1 agent '
                             'that is non-causal to the ego agent.')
    parser.add_argument('--scene_must_have_nc_all', action='store_true',
                        help='Scene must have at least 1 agent '
                             'that is non-causal to the ego agent.')
    parser.add_argument('--visualize', action='store_true',
                        help='Plot trajectory visualization')
    parser.add_argument('--visualize_agent_removal', action='store_true',
                        help='Whether to create visualization plots for agent removal.'
                             ' Those plots are time-consuming to generate.')
    parser.add_argument('--no_latex', action='store_true',
                        help='Whether not to use latex when creating plots.'
                             ' Some systems might not have a tex distribution installed.')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Frequency of log print')
    args = parser.parse_args()
    return args


def generate_scenes(args):
    trajectory_start_timestamp = 1  # TODO Move somewhere else. Also, why 1 and not 0? Would middle of trajectory be possibly better?

    # placeholder
    scenes = []
    tic = time.time()
    scene_idx = -1
    print("-" * 72)
    for i in tqdm(range(args.num_scenes)):
        num_ped = np.random.randint(args.min_num_ped, args.max_num_ped + 1)
        print(f"num_ped={num_ped}")
        scene_validity = False
        while not scene_validity:
            scene_idx += 1
            if args.simulator == 'orca':
                # Scene layout
                if args.layout == 'circle_crossing':
                    scene_configuration = generate_circle_crossing(num_ped, args.min_dist, args.scale)

                elif args.layout == "mixed_scenario_1":
                    scene_configuration = generate_mixed_scenario_1(num_ped, args.min_dist, args.scale, args.radius)

                elif args.layout == "mixed_scenario_2":
                    scene_configuration = None
                    while scene_configuration is None:
                        try:
                            scene_configuration = generate_mixed_scenario_2(num_ped, args.min_dist, args.scale,
                                                                            args.radius)
                        except RuntimeError as e:
                            print(f"Got a runtime error when calling `generate_mixed_scenario_2`: {e}")
                            print("Calling `generate_mixed_scenario_2` again")

                elif args.layout == "mixed_scenario_3":
                    scene_configuration = None
                    while scene_configuration is None:
                        try:
                            scene_configuration = generate_mixed_scenario_2(num_ped, args.min_dist, args.scale,
                                                                            args.radius, [0.0, 0.0, 0.2, 0.8, 0.0])
                        except RuntimeError as e:
                            print(f"Got a runtime error when calling `generate_mixed_scenario_2`: {e}")
                            print("Calling `generate_mixed_scenario_2` again")

                elif args.layout == "mixed_scenario_4":
                    scene_configuration = None
                    while scene_configuration is None:
                        try:
                            scene_configuration = generate_mixed_scenario_2(num_ped, args.min_dist, args.scale,
                                                                            args.radius, [0.0, 0.03, 0.07, 0.9, 0.0])
                        except RuntimeError as e:
                            print(f"Got a runtime error when calling `generate_mixed_scenario_2`: {e}")
                            print("Calling `generate_mixed_scenario_2` again")

                elif args.layout == "mixed_scenario_5":
                    scene_configuration = None
                    while scene_configuration is None:
                        scene_configuration = None
                        while scene_configuration is None:
                            try:
                                scene_configuration = generate_mixed_scenario_2(num_ped, args.min_dist, args.scale,
                                                                                args.radius, [0.0, 0.0, 0.8, 0.2, 0.0])
                            except RuntimeError as e:
                                print(f"Got a runtime error when calling `generate_mixed_scenario_2`: {e}")
                                print("Calling `generate_mixed_scenario_2` again")

                elif args.layout == "walkway":
                    scene_configuration = generate_walkway(num_ped, args.min_dist, args.scale, args.radius,
                                                           args.max_speed)
                else:
                    raise NotImplementedError

                # if scene_idx < 54:
                #     continue

                # Simulator
                orca_result = generate_orca_trajectory(
                    scene_configuration=scene_configuration,
                    radius=args.radius,
                    neighbordist=args.neighbor_dist,
                    horizon=args.horizon,
                    end_range=0.08,
                    fps=args.fps,
                    sampling_rate=args.fps // 2.5,
                    max_neighbors=10,
                    max_speed=args.max_speed,
                    max_step=args.max_length,
                    output_trajectory_start_timestamp=trajectory_start_timestamp,
                    output_trajectory_length=args.trajectory_length,
                    fov=args.fov,
                    very_close_ratio=args.very_close_ratio,
                    visibility_window=args.visibility_window,
                )
                scene_validity = orca_result["validity"]

                # Postprocessing simulator results
                labels = compute_estimated_causality_label(orca_result, causality_threshold=0.02)
                short_labels = compute_estimated_causality_label(orca_result, True, causality_threshold=0.02)
                orca_result["agent_type"] = orca_result["estimated_causality_label"] = labels
                orca_result["agent_type_shorthand"] = orca_result["estimated_causality_label_shorthand"] = short_labels
                orca_result["scenario_configuration"] = scene_configuration.to_dicts()
                orca_result["agent_setup_type"] = scene_configuration.get_setup_types()

                estimated_causality_label_counter = Counter(orca_result["estimated_causality_label_shorthand"])
                n_causal = estimated_causality_label_counter[estimated_causality_label['causal']['shorthand']]
                n_non_causal_ego = estimated_causality_label_counter[
                    estimated_causality_label['non_causal_ego']['shorthand']]
                n_non_causal_all = estimated_causality_label_counter[
                    estimated_causality_label['non_causal_all']['shorthand']]
                assert n_causal + n_non_causal_ego + n_non_causal_all + 1 == len(
                    orca_result["estimated_causality_label_shorthand"])

                scene_name = f"{args.simulator}" \
                             f"_idx={i:02d}" \
                             f"_iter={scene_idx:02d}" \
                             f"_num_ped={num_ped}" \
                             f"_DC={n_causal}" \
                             f"_NC-ego={n_non_causal_ego}" \
                             f"_NC-all={n_non_causal_all}"
                orca_result["scene_name"] = scene_name

                # Require the scene to have a certain degree of curvature for the ego trajectory
                orca_result["trajectory_interaction"] = estimate_interaction(trajectories=orca_result["trajectories"],
                                                                             dt=args.observation_length)
                orca_result["trajectory_curvature_mean"] = estimate_curvature(trajectories=orca_result["trajectories"],
                                                                              future_start_timestep=args.observation_length)
                orca_result["trajectory_curvature_max"] = estimate_curvature(trajectories=orca_result["trajectories"],
                                                                             future_start_timestep=args.observation_length,
                                                                             reduce_fn=np.max)

                if scene_validity and orca_result["trajectory_curvature_max"][0] < args.curvature_cutoff:
                    print(f"Invalid scene: Ego agent trajectory below curvature threshold "
                          f"(ego_curvature_max={orca_result['trajectory_curvature_max'][0]:.3f}"
                          f",cutoff={args.curvature_cutoff})"
                          f" (ego_curvature_mean={orca_result['trajectory_curvature_mean'][0]:.3f})"
                          f" (ego_trajectory_interaction={orca_result['trajectory_interaction'][0]:.3f})")
                    scene_validity = False

                # Require the scene to have at least one NC agent
                if scene_validity and args.scene_must_have_nc_ego and n_non_causal_ego == 0:
                    print(f"Invalid scene: None of {num_ped} agents were NC-ego.")
                    scene_validity = False
                if scene_validity and args.scene_must_have_nc_all and n_non_causal_all == 0:
                    print(f"Invalid scene: None of {num_ped} agents were NC-all.")
                    scene_validity = False

            else:
                raise NotImplementedError()

        nframes = len(orca_result["trajectories"][0])
        assert nframes == args.trajectory_length

        scene_trajectories = orca_result["trajectories"]
        scene_causality_labels = orca_result["causality_labels"]

        scenes.append(orca_result)

        assert len(scene_trajectories) == len(scene_causality_labels)
        assert len(scene_trajectories[0]) == len(scene_causality_labels[0])
        assert scene_causality_labels[0, :, 0].all()

        # debugging logs
        toc = time.time()
        print(f'Scene {i + 1}/{args.num_scenes} (seed={args.seed}) (dt={toc - tic:.2f})')
        tic = toc
        print(f"   nframes: {nframes}")
        print(f"   scene name: {scene_name}")
        print(f"   total agents: {len(scene_trajectories)} (ego + {len(scene_trajectories) - 1} other agents)")
        print(f"   agents causal (C) to ego agent: {n_causal}")
        print(f"   agents non-causal-ego (NC-ego) to ego agent: {n_non_causal_ego}")
        print(f"   agents non-causal-all (NC-all) to ego agent: {n_non_causal_all}")
        print("-" * 72)

        causal_in_any_timestep = np.array(scene_causality_labels).any(axis=-2)
        print(f"causal_in_any_timestep:")
        for agent_idx, c in enumerate(causal_in_any_timestep):
            print(f"\t{agent_idx}: {c}")

        print(f"causal effect (remove_agent_i_ade):")
        for agent_idx, ade in enumerate(orca_result["remove_agent_i_ade"]):
            print(f"\t{agent_idx}: {ade}")

        print(f"no indirect causal effect (based on ground truth causality graph):")
        for agent_idx, ice in enumerate(~orca_result["indirect_causality_effect"]):
            print(f"\t{agent_idx}: {ice}")

        print(f'trajectory curvature')
        for agent_idx, tc in enumerate(orca_result["trajectory_curvature_max"]):
            print(f"\t{agent_idx}: {tc}")

        print()

        # visualize
        if args.visualize:
            plot_trajectory(scene=scene_trajectories, figname=f'{args.logs_path}/figure/{scene_name}.png',
                            lim=args.scale + .5)
            animate(title=f"sv={orca_result['validity']}"
                          f" ego_curvature=[mean={orca_result['trajectory_curvature_mean'][0]:.3f}"
                          f",max={orca_result['trajectory_curvature_max'][0]:.3f}]"
                          f" ego_interaction={orca_result['trajectory_interaction'][0]:.3f}",
                    scene=orca_result["trajectories"], causality=orca_result["causality_labels"],
                    figname=f'{args.logs_path}/animation/{scene_name}.gif',
                    lim=args.scale + .5)
            if args.visualize_agent_removal:
                agent_removal_plots_path = os.path.join(args.logs_path, "agent_removal")
                ensure_dir(agent_removal_plots_path)
                plot_with_agent_removal(orca_result, agent_removal_plots_path, plot_id=scene_name,
                                        usetex=not args.no_latex)

    return scenes


def main(args):
    pl.seed_everything(args.seed, workers=True)
    scenes = generate_scenes(args)

    plot_causality_effect_per_agent_type(scenes, args.logs_path, usetex=not args.no_latex)
    plot_trajectory_curvature(scenes, args.logs_path, args.curvature_cutoff, usetex=not args.no_latex)

    dataset = {
        "scenes": scenes,
        "config": args,
    }

    dataset_output_pickle_path = f"{args.data_output_dir}/" \
                                 f"synthetic_dataset" \
                                 f"__{args.simulator}" \
                                 f"_layout-{args.layout}" \
                                 f"_ped-in-[{args.min_num_ped},{args.max_num_ped}]" \
                                 f"_scenes-{args.num_scenes}" \
                                 f"_r-{args.radius}" \
                                 f"_h-{args.horizon}" \
                                 f".pkl"
    with open(dataset_output_pickle_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"dataset_output_pickle_path:\n{os.path.abspath(dataset_output_pickle_path)}")


if __name__ == '__main__':
    args = parse_arguments()
    if args.data_output_dir is None:
        args.data_output_dir = args.logs_path
    ensure_dir(args.logs_path)
    ensure_dir(args.data_output_dir)
    redirect_stdout_and_stderr_to_file(os.path.join(args.logs_path, f"output__{get_str_formatted_time()}.txt"))

    nice_print(HORSE)
    print(f"args={args}")
    main(args)

    sys.stdout.flush()
    sys.stderr.flush()
