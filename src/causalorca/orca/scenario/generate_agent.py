from __future__ import annotations

import random
from typing import List

import numpy as np

from causalorca.orca.scenario.scenario_configuration import AgentStartConfiguration, SceneConfiguration
from causalorca.utils.util import Point, normalize


def _generate_random_agent_inside_square(
        agent_positions: List[Point],
        min_dist: float,
        square_center: Point,
        square_scale: float
):
    while True:
        px = square_center[0] + (random.random() - 0.5) * 2 * square_scale
        py = square_center[1] + (random.random() - 0.5) * 2 * square_scale
        if not _any_start_position_collision(Point(px, py), agent_positions, min_dist):
            break

    v = np.random.uniform(low=-1.0, high=1.0, size=2)
    dv = (v / np.linalg.norm(v)) * square_scale * 2
    goal_px = px + dv[0]
    goal_py = py + dv[1]

    vel = normalize(np.array([goal_py - px, goal_py - py]))

    return AgentStartConfiguration(Point(px, py), Point(goal_px, goal_py), Point(vel[0], vel[1]), "random_square")


def _any_start_position_collision(agent_pos: Point, other_agent_positions: List[Point], min_start_dist: float):
    assert min_start_dist > 0.0
    for other_agent_pos in other_agent_positions:
        start_position_distance = np.linalg.norm((agent_pos.x - other_agent_pos.x, agent_pos.y - other_agent_pos.y))
        if start_position_distance < min_start_dist:
            return True
    return False


def _generate_square_crossing_agent(
        agent_positions: List[Point],
        min_dist: float,
        square_center: Point,
        square_scale: float
):
    if np.random.random() > 0.5:
        sign = -1
    else:
        sign = 1

    while True:
        px = square_center[0] + np.random.random() * square_scale * sign
        py = square_center[1] + (np.random.random() - 0.5) * 2 * square_scale
        if not _any_start_position_collision(Point(px, py), agent_positions, min_dist):
            break
    while True:
        goal_px = square_center[0] + np.random.random() * square_scale * -sign
        goal_py = square_center[1] + (np.random.random() - 0.5) * 2 * square_scale
        if not _any_start_position_collision(Point(goal_px, goal_py), agent_positions, min_dist):
            break

    velocity = normalize(np.array([goal_py - px, goal_py - py]))

    # Move the goal further away so that the agent does not stop at the goal too early, since it might be unpredictable
    goal_px = px + velocity[0] * square_scale * 2
    goal_py = py + velocity[1] * square_scale * 2

    return AgentStartConfiguration(Point(px, py), Point(goal_px, goal_py), Point(velocity[0], velocity[1]),
                                   "square_crossing")


def _generate_agent_follower(agent_positions: List[Point], leader_id: int, leader_agent_config: AgentStartConfiguration,
                             distance, min_start_dist, noise=0.2, max_iter=50000):
    leader_pos = leader_agent_config.pos
    leader_goal = leader_agent_config.goal
    leader_vel = leader_agent_config.velocity

    for i in range(max_iter, -1, -1):
        pos_numpy = np.array(leader_pos) + normalize(np.array(leader_pos) - np.array(leader_goal)) * distance
        pos_numpy += np.array([random.uniform(0, noise), random.uniform(0, noise)])

        start = Point(pos_numpy[0], pos_numpy[1])
        goal = leader_goal
        vel = leader_vel

        if not _any_start_position_collision(start, agent_positions, min_start_dist):
            break

        if i == 0:
            raise RuntimeError(f"Could not generate setup after generating {max_iter} invalid setups.")

    return AgentStartConfiguration(start, goal, vel, "leader_follower", leader_id)


def _generate_parallel_overtake_agent(agent_positions, agent_goals, agent_to_overtake_config: AgentStartConfiguration,
                                      distance, min_start_and_goal_dist, noise=0.2, max_iter=5000):
    """
    Simulates the ParallelOvertake from socialforce:
    - https://www.svenkreiss.com/socialforce/scenarios.html
    - https://github.com/svenkreiss/socialforce/blob/c1dde8cc25979d68f67ede1bb77ce09a0406af0e/socialforce/scenarios.py#L43

    @param agent_to_overtake_config:
    @param distance:
    @param min_start_and_goal_dist:
    @param noise:
    @return:
    """
    start_to_goal = normalize(np.array(agent_to_overtake_config.goal) - np.array(agent_to_overtake_config.pos))
    for i in range(max_iter, -1, -1):
        # Noise
        u1 = (np.random.random(2) - 0.5) * 2 * noise

        start_np = np.array(agent_to_overtake_config.pos) - distance * start_to_goal + u1
        goal_np = np.array(agent_to_overtake_config.goal) + distance * start_to_goal + u1

        start = Point(start_np[0], start_np[1])
        goal = Point(goal_np[0], goal_np[1])

        if (
                not _any_start_position_collision(start, agent_positions, min_start_and_goal_dist)
                and
                not _any_start_position_collision(goal, agent_goals, min_start_and_goal_dist)
        ):
            break

        if i == 0:
            raise RuntimeError(f"Could not generate setup after generating {max_iter} invalid setups.")

    vel = agent_to_overtake_config.velocity

    return AgentStartConfiguration(start, goal, vel, "parallel_overtake")


def _generate_circle_crossing_agent(
        agent_positions: List[Point],
        min_dist: float,
        circle_center: Point,
        circle_radius: float,
        circle_angle_min_degrees: float = 0,
        circle_angle_max_degrees: float = 360,
):
    assert circle_angle_max_degrees >= circle_angle_min_degrees

    while True:
        angle = circle_angle_min_degrees + random.uniform(0, 1) * (circle_angle_max_degrees - circle_angle_min_degrees)
        angle_radians = angle * np.pi / 180
        px = circle_center.x + circle_radius * np.cos(angle_radians)
        py = circle_center.y + circle_radius * np.sin(angle_radians)
        if (
                not _any_start_position_collision(Point(px, py), agent_positions, min_dist)
                and
                not _any_start_position_collision(Point(-px, -py), agent_positions, min_dist)
        ):
            break

    start = Point(px, py)
    goal = Point(
        x=px - 2 * (px - circle_center.x),
        y=py - 2 * (py - circle_center.y)
    )
    velocity = normalize(np.array([goal.x - start.x, goal.y - start.y]))
    velocity = Point(velocity[0], velocity[1])

    return AgentStartConfiguration(start, goal, velocity, "circle_crossing"), angle


def _generate_static_agent(agent_positions, min_dist, width=4, height=8):
    # randomly initialize static objects in a square of (width, height)
    if np.random.random() > 0.5:
        sign = -1
    else:
        sign = 1
    while True:
        px = np.random.random() * width * 0.5 * sign
        py = (np.random.random() - 0.5) * height
        if not _any_start_position_collision(Point(px, py), agent_positions, min_dist):
            break

    return AgentStartConfiguration(Point(px, py), Point(px, py), Point(0, 0), "static")


def generate_circle_crossing(num_ped, min_dist, scale) -> SceneConfiguration:
    scene_configuration = SceneConfiguration()
    for _ in range(num_ped):
        circle_radius = scale
        agent_start_config, _ = _generate_circle_crossing_agent(
            agent_positions=scene_configuration.get_positions(),
            min_dist=min_dist,
            circle_center=Point(0, 0),
            circle_radius=circle_radius,
        )
        scene_configuration.agent_start_configurations += [scene_configuration]
    return scene_configuration
