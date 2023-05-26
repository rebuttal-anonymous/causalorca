from __future__ import annotations

from typing import List

import numpy as np

from causalorca.utils.util import Point


class AgentStartConfiguration:
    """
    Class to hold information about the initial configuration of an agent in the scene:
    initial position and velocity, initial goal, etc.
    """

    def __init__(self, pos: Point, goal: Point, velocity: Point, setup_type=None, leader=None, max_speed=None):
        self.pos = pos
        self.goal = goal
        self.velocity = velocity
        self.leader = leader
        self.max_speed = max_speed
        self.setup_type = setup_type

    def to_dict(self):
        return {k: v if type(v) != Point else tuple(v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d):
        return AgentStartConfiguration(**d)

    def copy(self):
        return AgentStartConfiguration(self.pos, self.goal, self.velocity, self.setup_type, self.leader, self.max_speed)

    @staticmethod
    def create_dummy():
        return AgentStartConfiguration(Point(np.nan, np.nan), Point(np.nan, np.nan), Point(np.nan, np.nan), "dummy")


class SceneConfiguration:
    """
    Class to hold the information about the initial scene setup, defined as a list of AgentStartConfiguration.
    """

    def __init__(self, agent_start_configurations: List[AgentStartConfiguration] = None):
        if agent_start_configurations is None:
            self.agent_start_configurations: List[AgentStartConfiguration] = list()
        else:
            self.agent_start_configurations: List[AgentStartConfiguration] = agent_start_configurations

    def to_dicts(self):
        """Convert to a list of dicts that can be easily serialized."""
        return [agent_config.to_dict() for agent_config in self.agent_start_configurations]

    @staticmethod
    def from_dicts(dicts) -> SceneConfiguration:
        """
        Create instance of the class made up by the provided list of dictionaries.

        @param dicts: Dictionaries created by calling the `SceneConfiguration.to_dicts` serialization method
        @return: An instance of `SceneConfiguration`
        """
        return SceneConfiguration([AgentStartConfiguration.from_dict(d) for d in dicts])

    def get_positions(self) -> List[Point]:
        return [a.pos for a in self.agent_start_configurations]

    def get_goals(self) -> List[Point]:
        return [a.goal for a in self.agent_start_configurations]

    def get_velocities(self) -> List[Point]:
        return [a.velocity for a in self.agent_start_configurations]

    def get_setup_types(self) -> List[Point]:
        return [a.setup_type for a in self.agent_start_configurations]

    def get_leaders(self) -> List:
        return [a.leader for a in self.agent_start_configurations]

    def get_max_speeds(self) -> List:
        return [a.max_speed for a in self.agent_start_configurations]

    def copy(self) -> SceneConfiguration:
        scene_configuration_copy = SceneConfiguration()
        scene_configuration_copy.agent_start_configurations = [a.copy() for a in self.agent_start_configurations]
        return scene_configuration_copy

    def remove_agent_at_idx(self, agent_idx) -> SceneConfiguration:
        self.agent_start_configurations.pop(agent_idx)

        # Since an agent was removed, the indices of the leader decrease by one
        for a in self.agent_start_configurations[agent_idx:]:
            if a.leader is None:
                continue

            if a.leader == agent_idx:
                # Leader had just been removed
                a.leader = None
            elif a.leader > agent_idx:
                a.leader -= 1

        return self

    def append(self, agent_start_configuration):
        self.agent_start_configurations += [agent_start_configuration]
        return self
