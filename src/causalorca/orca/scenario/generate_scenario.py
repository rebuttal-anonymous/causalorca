from __future__ import annotations

import random

import numpy as np

from causalorca.orca.scenario.generate_agent import _generate_circle_crossing_agent, \
    _generate_random_agent_inside_square, _generate_agent_follower, _generate_static_agent, \
    _generate_square_crossing_agent, _generate_parallel_overtake_agent, _any_start_position_collision
from causalorca.orca.scenario.scenario_configuration import SceneConfiguration, AgentStartConfiguration
from causalorca.utils.util import Point, normalize


def generate_mixed_scenario_1(n_agents, min_dist, scale, agent_radius,
                              agent_type_probabilities=(0.0, 0.65, 0.35, 0.0, 0.0, 0.0)) -> SceneConfiguration:
    scene_configuration = SceneConfiguration()

    # Add ego agent
    ego_start_config, ego_angle = _generate_circle_crossing_agent(
        agent_positions=scene_configuration.get_positions(),
        min_dist=min_dist,
        circle_center=Point(0, 0),
        circle_radius=scale,
    )
    scene_configuration.agent_start_configurations += [ego_start_config]

    # Add other agents
    agent_types = np.random.choice(
        a=["random_square", "random_circle", "random_circle_behind_ego", "random_circle_in_front_of_ego",
           "ego_follower", "leader_follower"],
        size=n_agents - 1,
        p=agent_type_probabilities,
        # p=[0.1, 0.6, 0.3, 0, 0.0, 0.0],
        # p=[0.0, 0.65, 0.35, 0.0, 0.0, 0.0],  # Mostly circle agents, with some agents behind ego
        # p=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Only ego followers
        # p=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Only random circle crossing
    )

    last_angle = ego_angle
    for agent_type in agent_types:
        if agent_type == "random_square":
            agent_start_config = _generate_random_agent_inside_square(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
                square_center=Point(0, 0),
                square_scale=scale,
            )

        elif agent_type == "random_circle":
            # circle_radius = scale
            circle_radius = random.uniform(scale * 0.8, scale * 1.2)
            agent_start_config, last_angle = _generate_circle_crossing_agent(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
                circle_center=Point(0, 0),
                circle_radius=circle_radius,
            )

        elif agent_type == "random_circle_behind_ego":
            angle = np.random.choice(a=[ego_angle, last_angle], p=[0.2, 0.8])
            circle_radius = random.uniform(scale * 1.4, scale * 1.6)
            agent_start_config, _ = _generate_circle_crossing_agent(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
                circle_center=Point(0, 0),
                circle_radius=circle_radius,
                circle_angle_min_degrees=angle - 20,
                circle_angle_max_degrees=angle + 20,
            )
            agent_start_config.setup_type += "_behind_ego"

        elif agent_type == "random_circle_in_front_of_ego":
            circle_radius = random.uniform(scale * 0.9, scale * 1.1)
            agent_start_config, _ = _generate_circle_crossing_agent(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
                circle_center=Point(ego_start_config.velocity.x * 3, ego_start_config.velocity.y * 3),
                circle_radius=circle_radius,
                circle_angle_min_degrees=ego_angle - 60,
                circle_angle_max_degrees=ego_angle + 60,
            )
            agent_start_config.setup_type += "_in_front_of_ego"

        elif agent_type == "ego_follower":
            agent_start_config = _generate_agent_follower(
                leader_id=0,
                agent_positions=scene_configuration.get_positions(),
                leader_agent_config=ego_start_config,
                distance=agent_radius * 3,
                min_start_dist=min_dist,
            )

        elif agent_type == "leader_follower":
            leader_id = 0
            while leader_id in scene_configuration.get_leaders():
                leader_id += 1

            agent_start_config = _generate_agent_follower(
                leader_id=leader_id,
                agent_positions=scene_configuration.get_positions(),
                leader_agent_config=scene_configuration.agent_start_configurations[leader_id],
                distance=agent_radius * 4,
                noise=agent_radius * 3,
                min_start_dist=agent_radius,
            )

        else:
            raise ValueError()

        scene_configuration.agent_start_configurations += [agent_start_config]

    return scene_configuration


def generate_mixed_scenario_2(n_agents, min_dist, scale, agent_radius,
                              agent_type_probability=(0.0, 0.0, 0.8, 0.2, 0.0)) -> SceneConfiguration:
    scene_configuration = SceneConfiguration()

    # First three agents are in the circle crossing
    # the rest humans will have a random starting and end position, or be static
    ## First agent, i.e., the ego agent
    ego_agent_config, ego_angle = _generate_circle_crossing_agent(
        agent_positions=scene_configuration.get_positions(),
        min_dist=min_dist,
        circle_center=Point(0, 0),
        circle_radius=scale,
    )
    scene_configuration.agent_start_configurations += [ego_agent_config]
    ## Second agent
    second_agent_config, second_angle = _generate_circle_crossing_agent(
        agent_positions=scene_configuration.get_positions(),
        min_dist=min_dist,
        circle_center=Point(0, 0),
        circle_radius=scale,
    )
    scene_configuration.agent_start_configurations += [second_agent_config]
    ## Third agent
    third_agent_config, third_angle = _generate_circle_crossing_agent(
        agent_positions=scene_configuration.get_positions(),
        min_dist=min_dist,
        circle_center=Point(0, 0),
        circle_radius=scale,
        circle_angle_min_degrees=ego_angle + 180 - 45,
        circle_angle_max_degrees=ego_angle + 180 + 45,
    )
    scene_configuration.agent_start_configurations += [third_agent_config]

    agent_types = np.random.choice(
        a=["static", "circle_crossing", "square_crossing", "leader_follower", "parallel_overtake"],
        size=n_agents - len(scene_configuration.agent_start_configurations),
        p=agent_type_probability,
    )

    for agent_type in agent_types:
        if agent_type == "static":
            agent_config = _generate_static_agent(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
            )

        elif agent_type == "circle_crossing":
            agent_config, _ = _generate_circle_crossing_agent(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
                circle_center=Point(0, 0),
                circle_radius=scale,
            )

        elif agent_type == "square_crossing":
            square_scale = scale
            agent_config = _generate_square_crossing_agent(
                agent_positions=scene_configuration.get_positions(),
                min_dist=min_dist,
                square_center=Point(0, 0),
                square_scale=square_scale,
            )
            # agent_config.max_speed = max_speed * 1.0

        elif agent_type == "leader_follower":
            leader_id = 0
            while leader_id in scene_configuration.get_leaders():
                leader_id += 1

            agent_config = _generate_agent_follower(
                leader_id=leader_id,
                agent_positions=scene_configuration.get_positions(),
                leader_agent_config=scene_configuration.agent_start_configurations[leader_id],
                distance=agent_radius * 4,
                noise=agent_radius * 3,
                min_start_dist=agent_radius,
            )

        elif agent_type == "parallel_overtake":
            # TODO If there are too many parallel overtakes on the on agent,
            #  it might be impossible to place in a way that the minimum distance is respected.
            #  We throw an error in that case, if after 5000 iterations a setup could not be found.
            agent_config = _generate_parallel_overtake_agent(
                agent_positions=scene_configuration.get_positions(),
                agent_goals=scene_configuration.get_goals(),
                agent_to_overtake_config=ego_agent_config,
                distance=agent_radius * 2,
                min_start_and_goal_dist=min_dist,
            )

        else:
            raise ValueError()

        scene_configuration.agent_start_configurations += [agent_config]

    return scene_configuration


def generate_walkway(n_agents, min_dist, scale, agent_radius, max_speed) -> SceneConfiguration:
    scene_configuration = SceneConfiguration()
    for agent_idx in range(n_agents):
        setup_type = "left" if agent_idx % 2 == 0 else "right"
        x_sign = -1 if setup_type == "left" else +1

        while True:
            # The noise, uniformly random in [-1,1]
            u1 = (random.random() - 0.5) * 2
            u2 = (random.random() - 0.5) * 2
            u3 = (random.random() - 0.5) * 2

            x_shift = agent_radius * 12 * u1
            y_start = agent_radius * 6 * u2
            # y_end = agent_radius * 8 * u3
            y_end = y_start + agent_radius * 4 * u3

            start = Point(x_sign * scale + x_shift, y_start)
            goal = Point(-x_sign * scale + x_shift, y_end)

            if (
                    not _any_start_position_collision(start, scene_configuration.get_positions(), min_dist)
                    and
                    not _any_start_position_collision(goal, scene_configuration.get_goals(), min_dist)
            ):
                break

        vel = normalize(np.array([goal.x - start.x, goal.y - start.y]))
        agent_start_config = AgentStartConfiguration(start, goal, vel, setup_type)
        scene_configuration.agent_start_configurations += [agent_start_config]

    return scene_configuration
