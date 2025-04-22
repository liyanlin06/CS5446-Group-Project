import sys
from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from luxai_s2.factory import compute_water_info
from lux.factory import Factory

from luxai_s2.state.state import State

class SimpleFactoryObservationWrapper(gym.ObservationWrapper):
    """
    Included features:
    - # of lichen tiles that can grow
    - # of connected lichen tiles
    - water storage
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(4,))

    def observation(self, obs):
        return SimpleFactoryObservationWrapper.convert_obs(obs, self.env.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    # should return an observation vector
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        lichen_map = shared_obs["board"]["lichen"]
        lichen_strains = shared_obs["board"]["lichen_strains"]

        factory_occupancy_map = np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        factories = dict()
        for agent in shared_obs["factories"]:
            for unit_id in shared_obs["factories"][agent]:
                f_data = shared_obs["factories"][agent][unit_id]
                factory = Factory(**f_data, env_cfg=env_cfg)
                factory_occupancy_map[factory.pos_slice] = factory.strain_id
        
        for agent in obs.keys():
            obs_vec = np.zeros(
                4,
            )

            factories = shared_obs["factories"][agent]

            lichen_vec = np.zeros(2)
            cargo_vec = np.zeros(1)
            steps_vec = np.zeros(1)

            forbidden = (
                (shared_obs["board"]["rubble"] > 0)
                | (factory_occupancy_map != -1)
                | (shared_obs["board"]["ice"] > 0)
                | (shared_obs["board"]["ore"] > 0)
            )
            deltas = [
                np.array([0, -2]),
                np.array([-1, -2]),
                np.array([1, -2]),
                np.array([0, 2]),
                np.array([-1, 2]),
                np.array([1, 2]),
                np.array([2, 0]),
                np.array([2, -1]),
                np.array([2, 1]),
                np.array([-2, 0]),
                np.array([-2, -1]),
                np.array([-2, 1]),
            ]
            for k, factory in factories.items():
                # here we track a normalized position of the first friendly factory
                grow_lichen_positions, connected_lichen_positions = compute_water_info(
                    np.stack(deltas) + factory["pos"],
                    env_cfg.MIN_LICHEN_TO_SPREAD,
                    lichen_map,
                    lichen_strains,
                    factory_occupancy_map,
                    factory["strain_id"],
                    forbidden
                )
                lichen_vec[0] = len(grow_lichen_positions)
                lichen_vec[1] = len(connected_lichen_positions)
                lichen_vec /= env_cfg.map_size
                cargo_vec[0] = factory["cargo"]["water"] / 2000 # 2000 is a hyperparameter
                break
        

            if hasattr(env_cfg, "max_episode_steps"):
                env_steps = shared_obs["real_env_steps"] / env_cfg.max_episode_steps
            else:
                env_steps = shared_obs["real_env_steps"] / env_cfg.max_episode_length
            steps_vec[0] = env_steps
            obs_vec = np.concatenate(
                [lichen_vec, cargo_vec, steps_vec], axis=-1
            )
            observation[agent] = obs_vec

        return observation


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(15,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    # should return an observation vector
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ore_map = shared_obs["board"]["ore"]
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_tile_locations = np.argwhere(ore_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                15,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                # compute closest ore tile
                ore_tile_distances = np.mean(
                    (ore_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ore tile location
                closest_ore_tile = (
                    ore_tile_locations[np.argmin(ore_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos, closest_ore_tile - pos], axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation
