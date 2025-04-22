"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import random
import os.path as osp
import sys
from pprint import pprint
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper
from wrappers import SimpleFactoryDiscreteController, SimpleFactoryObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
# UNITS_MODEL_WEIGHTS_RELATIVE_PATH = "./units_best_model"
UNITS_MODEL_WEIGHTS_RELATIVE_PATH = "logs/exp_1/models/units/best_model"
# FACTORIES_MODEL_WEIGHTS_RELATIVE_PATH = "./factories_best_model"
FACTORIES_MODEL_WEIGHTS_RELATIVE_PATH = "logs/exp_1/models/factories/best_model.zip"

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.units_policy = PPO.load(osp.join(directory, UNITS_MODEL_WEIGHTS_RELATIVE_PATH))
        self.factories_policy = PPO.load(osp.join(directory, FACTORIES_MODEL_WEIGHTS_RELATIVE_PATH))

        self.units_controller = SimpleUnitDiscreteController(self.env_cfg)
        self.factories_controller = SimpleFactoryDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # CS5446: here create factories with metal=150 and water=150 until we run out of resources
        # if obs["teams"][self.player]["metal"] < 150 or obs["teams"][self.player]["water"] < 150:
        #     return dict()
        if obs["teams"][self.player]["metal"] < 500 or obs["teams"][self.player]["water"] < 500:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False
        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        return dict(spawn=pos, metal=150, water=150)
    
    def _act(self, obs, convert_obs, controller, policy):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            # print(self.policy.policy, file=sys.stderr)
            # exit(1)
            # print("obs device:", obs.device, file=sys.stderr)
            # print("model device:", next(self.policy.policy.parameters()).device, file=sys.stderr)
            device = next(policy.parameters()).device
            obs = obs.to(device)
            features = policy.features_extractor(obs.unsqueeze(0))
            x = policy.mlp_extractor.shared_net(features)
            logits = policy.action_net(x) # shape (1, N) where N=12 for the default controller

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)
            # print(actions[0], file=sys.stderr)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        return lux_action
    
    # def mask_obs_for_unit(self, obs, unit_id):
    #     # Create a copy of the observation
    #     masked_obs = obs.copy()
    #     # Keep only the specific unit
    #     masked_obs["units"] = {self.player: {unit_id: obs["units"][self.player].get(unit_id, {})}}
    #     # Remove opponent units
    #     masked_obs["units"][self.opp_player] = {}

    #     return masked_obs

    # def mask_obs_for_factory(self, obs, factory_id):
    #      # Create a copy of the observation
    #     masked_obs = obs.copy()
    #     # Keep only the specific factory
    #     masked_obs["factories"] = {self.player: {factory_id: obs["factories"][self.player].get(factory_id, {})}}
    #     # Remove opponent factories
    #     masked_obs["factories"][self.opp_player] = {}

    #     return masked_obs

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # CS5446: implementation of multiple units and factories
        unit_action = dict()
        factory_action = dict()

        units = obs["units"][self.player]
        factories = obs["factories"][self.player]

        # for unit_id, unit in units.items():
        #     # Create a masked observation for the unit
        #     masked_obs = self.mask_obs_for_unit(obs, unit_id)
        #     # Apply the unit policy to the masked observation
        #     unit_action.update(self._act(masked_obs, SimpleUnitObservationWrapper.convert_obs, self.units_controller, self.units_policy.policy))

        # for factory_id, factory in factories.items():
        #     # Create a masked observation for the factory
        #     masked_obs = self.mask_obs_for_factory(obs, factory_id)
        #     # Apply the factory policy to the masked observation
        #     factory_action.update(self._act(masked_obs, SimpleFactoryObservationWrapper.convert_obs, self.factories_controller, self.factories_policy.policy))

        # optional: not doing observation masking
        unit_action = self._act(
            obs, SimpleUnitObservationWrapper.convert_obs, self.units_controller, self.units_policy.policy
        )
        factory_action = self._act(
                obs, SimpleFactoryObservationWrapper.convert_obs, self.factories_controller, self.factories_policy.policy
        )

        # # rule-based overrides
        # for unit_id, unit in units.items():
            # apply the unit policy output to all units using a rule-based approach
            # if unit_id not in unit_action and len(unit_action) != 0:
            #     unit_action[unit_id] = next(iter(unit_action.values()))

            # move away from spawn point
            # factories_pos = [factory['pos'] for factory in factories.values()]
            # if any(np.array_equal(unit['pos'], factory_pos) for factory_pos in factories_pos):
            #     unit_action[unit_id] = [np.array([0, random.randint(1, 4), 0, 0, 0, 1])]

        for factory_id, factory in factories.items():
            # build troops (1 heavy robot with n light robots) in early stages
            # if factory["cargo"]["metal"] >= 10 and step < 30 and 1 not in factory_action.values():
            #     factory_action[factory_id] = 0  # Spawn a light robot

            # apply the factory policy output to all factories using a rule-based approach
            if factory_id not in factory_action and len(factory_action) != 0:
                factory_action[factory_id] = next(iter(factory_action.values()))

        # # Override factory policy for watering
        updated_factory_action = dict()
        for factory_id, action in factory_action.items():
            if action == 2:  # Action 2 corresponds to watering
                # Calculate the probability of watering based on the current step
                if (step / 1000) ** 3 > np.random.rand():
                    updated_factory_action[factory_id] = action
            else:
                updated_factory_action[factory_id] = action


        # if factory_id not in factory_action and len(factory_action) != 0:
        #     factory_action[factory_id] = next(iter(factory_action.values()))

        lux_action = dict()
        lux_action.update(unit_action)
        lux_action.update(updated_factory_action)

        # print(lux_action, file=sys.stderr)
        # if step > 1000:
        # for unit_id in factories.keys():
        #     factory = factories[unit_id]
        #     if 1000 - step < 50 and factory["cargo"]["water"] > 100:
        #         lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
