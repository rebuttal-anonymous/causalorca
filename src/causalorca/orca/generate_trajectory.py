from __future__ import annotations

from collections import namedtuple

import networkx as nx
import numpy as np
import rvo2

from causalorca.orca.scenario.scenario_configuration import SceneConfiguration, AgentStartConfiguration
from causalorca.utils.util import is_smooth


def causal_vector(num_agent, num_ped, sim):
    causal = np.zeros(num_ped)
    causal[num_agent] = 1
    nb_neigh = sim.getAgentNumAgentNeighbors(num_agent)

    for j in range(nb_neigh):
        neighbor = sim.getAgentAgentNeighbor(num_agent, j)
        causal[neighbor] = 1

    return causal


def _simulate_trajectories(scene_configuration: SceneConfiguration, simulation_config,
                           output_trajectory_start_timestamp, output_trajectory_length):
    initial_positions = scene_configuration.get_positions()
    initial_velocities = scene_configuration.get_velocities()
    goals = scene_configuration.get_goals()
    leaders = scene_configuration.get_leaders()
    max_speeds = scene_configuration.get_max_speeds()

    num_ped = len(scene_configuration.agent_start_configurations)
    assert num_ped == len(goals) == len(initial_velocities) == len(leaders) == len(max_speeds)

    # Initialize trajectories and causality labels
    trajectories = [[pos] for pos in initial_positions]
    causality_labels = [[] for _ in initial_positions]
    for i in range(num_ped):
        causal = np.zeros(num_ped)
        causal[i] = 1
        causality_labels[i].append(causal)

    # Create simulator
    sim = rvo2.PyRVOSimulator(*simulation_config["simulator_args"])
    for i in range(num_ped):
        pos = (initial_positions[i][0], initial_positions[i][1])
        vel = (initial_velocities[i][0], initial_velocities[i][1])
        sim.addAgent(pos, *simulation_config["common_agent_args"], vel)
        if max_speeds[i] is not None:
            sim.setAgentMaxSpeed(i, max_speeds[i])

    # Perform simulation
    done = False
    cnt_step = 0
    while not done and cnt_step < simulation_config["max_step"]:
        # simulate step
        cnt_step += 1
        # noting down the causality label before doing the step, so that the causality label says whether
        # an agent was causal to another agent when the current step was made (not after the step was made)
        if cnt_step % simulation_config["sampling_rate"] == 0:
            for i in range(num_ped):
                causal = causal_vector(i, num_ped, sim)
                causality_labels[i].append(causal)
        sim.doStep()

        reaching_goal = []

        for i in range(num_ped):
            if max_speeds[i] is None:
                max_speeds[i] = sim.getAgentMaxSpeed(i)

        for i in range(num_ped):

            # update memory
            position = sim.getAgentPosition(i)
            if cnt_step % simulation_config["sampling_rate"] == 0:
                trajectories[i].append(position)

            # check if this agent stops
            if leaders[i] is not None:
                leader_position = sim.getAgentPosition(leaders[i])
                goals[i] = leader_position

            if np.linalg.norm(np.array(position) - np.array(goals[i])) < simulation_config["simulator_args"].radius:
                reaching_goal.append(True)
                sim.setAgentMaxSpeed(i, 0.0)
                sim.setAgentPrefVelocity(i, (0, 0))
            else:
                reaching_goal.append(False)
                sim.setAgentMaxSpeed(i, max_speeds[i])
                velocity = np.array((goals[i][0] - position[0], goals[i][1] - position[1]))
                speed = np.linalg.norm(velocity)
                pref_vel = 1 * velocity / speed if speed > 1 else velocity
                sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))

        # done = all(reaching_goal) # why not all ???
        done = reaching_goal[0]

    # Output only subset of timesteps
    assert np.array(trajectories).shape[:2] == np.array(causality_labels).shape[:2]
    output_trajectory_timestep_range = range(output_trajectory_start_timestamp,
                                             output_trajectory_start_timestamp + output_trajectory_length)
    output_trajectories = np.array(trajectories)[:, output_trajectory_timestep_range, :]
    output_causality_labels = np.array(causality_labels)[:, output_trajectory_timestep_range, :]

    smooth = is_smooth(output_trajectories)
    if smooth:
        valid = True
    else:
        valid = False
        print("Warning: invalid trajectories (done={}, smooth={})".format(done, smooth))
        # animate(scene=np.array(trajectories), causality=np.array(causality_labels),
        #         figname=f'{args.logs_path}/animation_nonfiltered/{get_str_formatted_time()}.gif',
        #         lim=args.scale + .5)

    return output_trajectories, output_causality_labels, valid


def _compute_ade(traj1, traj2):
    return np.mean(np.linalg.norm(traj1 - traj2, axis=-1), axis=-1)


def _estimate_causal_effect_of_removing_agent_i(
        clean_trajectories, output_trajectory_start_timestamp, output_trajectory_length,
        agent_index, simulation_step_to_remove_at, scene_configuration: SceneConfiguration, simulation_config, ):
    assert simulation_step_to_remove_at == 0
    assert len(scene_configuration.agent_start_configurations) > 1

    scene_configuration_without_agent_i = scene_configuration.copy().remove_agent_at_idx(agent_index)

    trajectories, causality_labels, validity = _simulate_trajectories(
        scene_configuration=scene_configuration_without_agent_i,
        simulation_config=simulation_config,
        output_trajectory_start_timestamp=output_trajectory_start_timestamp,
        output_trajectory_length=output_trajectory_length,
    )

    trajectories = np.insert(trajectories, agent_index, np.full_like(trajectories[0], np.nan), axis=0)
    causality_labels = np.insert(causality_labels, agent_index, np.full_like(causality_labels[0], np.nan), axis=0)
    assert len(trajectories) == len(causality_labels) == len(clean_trajectories)

    ade = [_compute_ade(traj, clean_traj) if traj is not None else None
           for traj, clean_traj
           in zip(trajectories, clean_trajectories)]
    return trajectories, causality_labels, validity, ade


def generate_orca_trajectory(scene_configuration: SceneConfiguration,
                             radius=1.0, neighbordist=5, horizon=1.0,
                             end_range=0.01, fps=50, sampling_rate=1,
                             max_neighbors=10, max_speed=1.0, max_step=10000,
                             output_trajectory_start_timestamp=1, output_trajectory_length=20,
                             fov=135, very_close_ratio=0.1, visibility_window=5):
    # Simulator parameters description: https://gamma.cs.unc.edu/RVO2/documentation/2.0/params.html
    SimulatorArgs = namedtuple("SimulatorArgs",
                               ["timeStep",
                                "neighborDist", "maxNeighbors",
                                "timeHorizon", "timeHorizonObst",
                                "radius", "maxSpeed", "velocity",
                                "fov", "veryCloseRatio", "neighborVisibilityWindowSize"])
    CommonAgentArgs = namedtuple("CommonAgentArgs",
                                 ["neighborDist", "maxNeighbors",
                                  "timeHorizon", "timeHorizonObst",
                                  "radius", "maxSpeed"])
    simulator_args = SimulatorArgs(1.0 / fps, neighbordist, max_neighbors, horizon, horizon, radius, max_speed,
                                   (0, 0), fov, very_close_ratio, visibility_window * sampling_rate)
    common_agent_args = CommonAgentArgs(neighbordist, max_neighbors, horizon, horizon, radius, max_speed)
    simulation_config = {
        "common_agent_args": common_agent_args,
        "simulator_args": simulator_args,
        "end_range": end_range,
        "max_step": max_step,
        "sampling_rate": sampling_rate,
    }

    # Perform simulation with all agents
    trajectories, causality_labels, validity = _simulate_trajectories(
        scene_configuration=scene_configuration,
        simulation_config=simulation_config,
        output_trajectory_start_timestamp=output_trajectory_start_timestamp,
        output_trajectory_length=output_trajectory_length,
    )

    # Compute if an indirect effect exists in the causality graph of the scene
    indirect_causality_effect = _compute_existance_of_indirect_causality_effect(causality_labels)

    # Estimate the causal effect of removing one agent
    remove_agent_i_trajectories = []
    remove_agent_i_causality_labels = []
    remove_agent_i_validity = []
    remove_agent_i_ade = []
    for agent_idx in range(len(scene_configuration.agent_start_configurations)):
        simulation_step_to_remove_at = 0  # TODO
        trajectories_i, causality_labels_i, validity_i, ade_i = _estimate_causal_effect_of_removing_agent_i(
            clean_trajectories=trajectories,
            output_trajectory_start_timestamp=output_trajectory_start_timestamp,
            output_trajectory_length=output_trajectory_length,
            agent_index=agent_idx,
            simulation_step_to_remove_at=simulation_step_to_remove_at,
            scene_configuration=scene_configuration,
            simulation_config=simulation_config,
        )
        remove_agent_i_trajectories += [trajectories_i]
        remove_agent_i_causality_labels += [causality_labels_i]
        remove_agent_i_validity += [validity_i]
        remove_agent_i_ade += [ade_i]
    remove_agent_i_trajectories = np.array(remove_agent_i_trajectories)
    remove_agent_i_causality_labels = np.array(remove_agent_i_causality_labels)
    remove_agent_i_validity = np.array(remove_agent_i_validity)
    remove_agent_i_ade = np.array(remove_agent_i_ade)

    # Sanity check that two simulations give the same output
    trajectories_sanity_check, causality_labels_sanity_check, validity_sanity_check, ade_sanity_check = _estimate_causal_effect_of_removing_agent_i(
        clean_trajectories=np.insert(trajectories, -1, np.full_like(trajectories[0], np.nan), axis=0),
        output_trajectory_start_timestamp=output_trajectory_start_timestamp,
        output_trajectory_length=output_trajectory_length,
        agent_index=len(trajectories),
        simulation_step_to_remove_at=0,
        scene_configuration=scene_configuration.copy().append(AgentStartConfiguration.create_dummy()),
        simulation_config=simulation_config,
    )
    assert np.array_equal(trajectories, trajectories_sanity_check[:-1])
    assert np.array_equal(causality_labels, causality_labels_sanity_check[:-1])
    assert validity == validity_sanity_check

    result = {
        # original simulation data
        "trajectories": trajectories,  # np.array of shape (n_agents, n_timesteps, n_coordinates=2)
        "causality_labels": causality_labels,  # was x directly causal to y at t? np.array of shape (y,t,x), TO->FROM
        "validity": validity,

        # remove agent simulation data
        "remove_agent_i_trajectories": remove_agent_i_trajectories,
        "remove_agent_i_causality_labels": remove_agent_i_causality_labels,
        "remove_agent_i_validity": remove_agent_i_validity,

        # causal effect estimated by ADE
        "remove_agent_i_ade": remove_agent_i_ade,  # causal effect from x to y in ade, np.array of shape (x,y), FROM->TO

        # indirect causal effect ground truth, based on causality graph
        "indirect_causality_effect": indirect_causality_effect,  # was x indirectly causal to y? (x,y), FROM->TO
    }

    return result


def _compute_existance_of_indirect_causality_effect(scene_causality_labels):
    """
    TODO document.
    If there is a path from one positions of node x to positions of node y in the causal graph,
    then node x is causal to node y.

    @param scene_causality_labels:
    @return:
    """
    indirect_causality_effects = []
    edge_list = []
    for from_agent_idx in range(scene_causality_labels.shape[2]):
        for timestep in range(scene_causality_labels.shape[1]):
            for to_agent_idx in range(scene_causality_labels.shape[0]):
                if scene_causality_labels[to_agent_idx][timestep][from_agent_idx] != 1.0:
                    continue
                source, target = (from_agent_idx, timestep - 1), (to_agent_idx, timestep)
                edge_list += [(source, target)]

    dg = nx.DiGraph()
    dg.add_edges_from(edge_list)
    assert len(dg.edges()) == scene_causality_labels.sum()

    def has_path(from_agent_idx, to_agent_idx):
        for t1 in range(scene_causality_labels.shape[1]):
            for t2 in range(scene_causality_labels.shape[1]):
                source, target = (from_agent_idx, t1 - 1), (to_agent_idx, t2)
                if source not in dg.nodes() or target not in dg.nodes():
                    continue
                if nx.has_path(dg, source, target):
                    return True
        return False

    for from_agent_idx in range(scene_causality_labels.shape[2]):
        indirect_causality_effect = []
        for to_agent_idx in range(scene_causality_labels.shape[0]):
            if from_agent_idx == to_agent_idx:
                indirect_causality_effect += [True]
            else:
                indirect_causality_effect += [has_path(from_agent_idx, to_agent_idx)]
        indirect_causality_effects += [indirect_causality_effect]

    return np.array(indirect_causality_effects)
