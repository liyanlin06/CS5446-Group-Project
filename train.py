"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""


import copy
import os.path as osp
from pprint import pprint
import gc

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from gym.wrappers import TimeLimit
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.utils.heuristics.bidding import zero_bid
from wrappers import SB3Wrapper
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper
from wrappers import SimpleFactoryDiscreteController, SimpleFactoryObservationWrapper

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import configure

class CustomUnitsEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.policy = None # factories policy
        self.units_controller = SimpleUnitDiscreteController(self.env.env_cfg)
        self.factories_controller = SimpleFactoryDiscreteController(self.env.env_cfg)
        self.action_space = self.units_controller.action_space

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 2000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 500000

        # units act based on old policy
        prev_obs = self.env.env.prev_obs
        raw_obs = prev_obs
        factories_obs = SimpleFactoryObservationWrapper.convert_obs(
            raw_obs, self.env.env_cfg
        )
        factories_obs = factories_obs[agent]
        factories_obs = th.from_numpy(factories_obs).float()
        with th.no_grad():
            action_mask = (
                th.from_numpy(self.factories_controller.action_masks(agent, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            device = next(self.policy.parameters()).device
            factories_obs = factories_obs.to(device)
            features = self.policy.features_extractor(factories_obs.unsqueeze(0))
            x = self.policy.mlp_extractor.shared_net(features)
            logits = self.policy.action_net(x) # shape (1, N) where N=2 for the default controller

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)
        
        lux_action = {
            agent: dict(),
            opp_agent: dict(),
        }

        for unit_id in prev_obs[agent]["units"][agent].keys():
            lux_action[agent].update(self.units_controller.action_to_lux_action(
                agent=agent, obs=prev_obs, action=action
            ))
        
        for factory_id in prev_obs[agent]["factories"][agent].keys():
            lux_action[agent].update(self.factories_controller.action_to_lux_action(
                agent=agent, obs=prev_obs, action=actions[0]
            ))
        
        for opp_factory_id in prev_obs[opp_agent]["factories"][opp_agent].keys():
            lux_action[opp_agent].update(self.factories_controller.action_to_lux_action(
                agent=opp_agent, obs=prev_obs, action=0 # water
            ))
        
        obs, _, done, info = self.env.env.step(lux_action)
        obs = SimpleUnitObservationWrapper.convert_obs(
            obs, self.env.env_cfg
        )
        obs = obs[agent]
        done = done[agent]
        
        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["ore_dug"] = (
            stats["generation"]["ore"]["HEAVY"] + stats["generation"]["ore"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["metal_produced"] = stats["generation"]["metal"]

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        # CS5446
        # track more metrics to determine which is good
        metrics["power_used"] = stats.get("power_used", 0)

        # Track power metrics
        if self.prev_step_metrics is not None:
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
            water_produced_this_step = metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            metal_produced_this_step = metrics["metal_produced"] - self.prev_step_metrics["metal_produced"]
            power_used_this_step = metrics["power_used"] - self.prev_step_metrics.get("power_used", 0)
            
            # Power efficiency
            if power_used_this_step > 0:
                resources_gained = (ice_dug_this_step + ore_dug_this_step + 
                                   water_produced_this_step*100 + metal_produced_this_step*100)
                metrics["power_efficiency"] = resources_gained / power_used_this_step
            else:
                metrics["power_efficiency"] = 0
            
            # Resource collection balance
            total_resources = ice_dug_this_step + ore_dug_this_step
            if total_resources > 0:
                metrics["ice_ratio"] = ice_dug_this_step / total_resources
            else:
                metrics["ice_ratio"] = 0
            
            # Infer resource transfers (approximation)
            expected_water = ice_dug_this_step / 10  # Assuming 10 ice -> 1 water conversion rate
            if water_produced_this_step > 0 and expected_water > 0:
                metrics["transfer_efficiency"] = min(1.0, water_produced_this_step / expected_water)
            else:
                metrics["transfer_efficiency"] = 0

        # Track unit metrics
        metrics["unit_count"] = len(self.env.state.units[agent])
        if hasattr(self, 'prev_unit_count'):
            units_lost = max(0, self.prev_unit_count - metrics["unit_count"])
            metrics["units_lost"] = units_lost
        else:
            metrics["units_lost"] = 0
        self.prev_unit_count = metrics["unit_count"]


        # Track resource locations
        ice_locations = np.argwhere(self.env.state.board.ice == 1)
        if len(ice_locations) > 0:
            metrics["ice_tiles_remaining"] = len(ice_locations)
        else:
            metrics["ice_tiles_remaining"] = 0

        ore_locations = np.argwhere(self.env.state.board.ore == 1)
        if len(ore_locations) > 0:
            metrics["ore_tiles_remaining"] = len(ore_locations)
        else:
            metrics["ore_tiles_remaining"] = 0

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        
        # CS5446
        # if factory is destroyed then -1000
        # factories_left = len(self.env.state.factories[agent])
        # if factories_left == 0:
        #     reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
            water_produced_this_step = (
                metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            metal_produced_this_step = (
                metrics["metal_produced"] - self.prev_step_metrics["metal_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step
            # reward += (ore_dug_this_step / 100 + metal_produced_this_step) * 0.1

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        self.prev_obs = obs
        return obs
    
    def init_old_policy(self, obs_space, act_space, policy_kwargs):
        self.policy = MlpPolicy(
            obs_space,
            act_space,
            lr_schedule=lambda _: 3e-4,
            **policy_kwargs,
        )
    
    def set_old_policy(self, policy_state_dict):
        self.policy.load_state_dict(policy_state_dict)

class CustomFactoriesEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.policy = None # units policy
        self.units_controller = SimpleUnitDiscreteController(self.env.env_cfg)
        self.factories_controller = SimpleFactoryDiscreteController(self.env.env_cfg)
        self.action_space = self.factories_controller.action_space

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 2000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 500000

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below

        # units act based on old policy
        prev_obs = self.env.env.prev_obs
        raw_obs = prev_obs
        units_obs = SimpleUnitObservationWrapper.convert_obs(
            raw_obs, self.env.env_cfg
        )
        units_obs = units_obs[agent]
        units_obs = th.from_numpy(units_obs).float()
        with th.no_grad():
            action_mask = (
                th.from_numpy(self.units_controller.action_masks(agent, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            device = next(self.policy.parameters()).device
            units_obs = units_obs.to(device)
            features = self.policy.features_extractor(units_obs.unsqueeze(0))
            x = self.policy.mlp_extractor.shared_net(features)
            logits = self.policy.action_net(x) # shape (1, N) where N=16 for the default controller

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)
        
        lux_action = {
            agent: dict(),
            opp_agent: dict(),
        }

        for unit_id in prev_obs[agent]["units"][agent].keys():
            lux_action[agent].update(self.units_controller.action_to_lux_action(
                agent=agent, obs=prev_obs, action=actions[0]
            ))
        
        for factory_id in prev_obs[agent]["factories"][agent].keys():
            lux_action[agent].update(self.factories_controller.action_to_lux_action(
                agent=agent, obs=prev_obs, action=action
            ))
        
        for opp_factory_id in prev_obs[opp_agent]["factories"][opp_agent].keys():
            lux_action[opp_agent].update(self.factories_controller.action_to_lux_action(
                agent=opp_agent, obs=prev_obs, action=0 # water
            ))
        
        obs, _, done, info = self.env.env.step(lux_action)
        obs = SimpleFactoryObservationWrapper.convert_obs(
            obs, self.env.env_cfg
        )
        obs = obs[agent]
        done = done[agent]

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        # print(action)
        # print(lux_action)
        # fac = self.env.state.factories[opp_agent]['factory_1']
        # x_slice = slice(fac.pos.x - 2, fac.pos.x + 3)
        # y_slice = slice(fac.pos.y - 2, fac.pos.y + 3)
        # print(self.env.state.board.factory_occupancy_map[x_slice, y_slice])
        # print(self.env.state.board.rubble[x_slice, y_slice])
        # print(self.env.state.board.ice[x_slice, y_slice])
        # print(self.env.state.board.ore[x_slice, y_slice])
        # print(self.env.state.board.lichen[x_slice, y_slice])
        # print(fac.grow_lichen_positions)
        # print(fac.connected_lichen_positions)
        # print(self.env.state.stats[agent]['generation']['lichen'])
        # print(self.env.state.board.lichen.sum())

        info = dict()
        metrics = dict()
        metrics["lichen_grown"] = stats["generation"]["lichen"]
        metrics["lichen_diff"] = (
            stats["generation"]["lichen"] - self.env.state.stats[opp_agent]["generation"]["lichen"]
        )
        metrics["power_factory_produced"] = stats["generation"]["power"]["FACTORY"]

       
        # CS5446
        # track components of strategic reward for evaluation
        factories_left = len(self.env.state.factories[agent])

        if factories_left > 0:
            # Initialize totals
            connected_lichen_total = 0
            growth_potential_total = 0
            surrounding_ice_total = 0
            
            # Calculate totals
            for factory_id, factory in self.env.state.factories[agent].items():
                connected_lichen_total += len(factory.connected_lichen_positions)
                growth_potential_total += len(factory.grow_lichen_positions)
                
                x_slice = slice(factory.pos.x - 2, factory.pos.x + 3)
                y_slice = slice(factory.pos.y - 2, factory.pos.y + 3)
                surrounding_ice_total += self.env.state.board.ice[x_slice, y_slice].sum()
            
            # Add only the averages to metrics
            metrics["connected_lichen_avg"] = connected_lichen_total / factories_left
            metrics["growth_potential_avg"] = growth_potential_total / factories_left
            metrics["surrounding_ice_avg"] = surrounding_ice_total / factories_left
        else:
            # Set to zero if no factories
            metrics["connected_lichen_avg"] = 0
            metrics["growth_potential_avg"] = 0
            metrics["surrounding_ice_avg"] = 0

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        
        # CS5446
        # if factory is destroyed then -1000
        if factories_left == 0:
            reward = -1000
        elif self.prev_step_metrics is not None:
            # Primary reward: Lichen growth
            lichen_grown_this_step = metrics["lichen_grown"] - self.prev_step_metrics["lichen_grown"]
            
            # Net power production
            power_produced_this_step = (
                metrics["power_factory_produced"] - self.prev_step_metrics["power_factory_produced"] - self.env.env_cfg.FACTORY_CHARGE
            )
            
            # Additional strategic rewards (experimental)
            strategic_reward = 0
            for factory_id, factory in self.env.state.factories[agent].items():
                # Connectivity reward
                connected_lichen_reward = len(factory.connected_lichen_positions) / 100
                
                # Growth potential reward
                growth_potential_reward = len(factory.grow_lichen_positions) / 100
                
                # Area analysis
                x_slice = slice(factory.pos.x - 2, factory.pos.x + 3)
                y_slice = slice(factory.pos.y - 2, factory.pos.y + 3)
                
                # Resource proximity
                surrounding_ice = self.env.state.board.ice[x_slice, y_slice].sum() / 10
                
                strategic_reward += connected_lichen_reward + growth_potential_reward + surrounding_ice
            
            # Normalize by number of factories
            if factories_left > 0:
                strategic_reward /= factories_left
            
            # Combined reward with appropriate weights
            test_reward = (lichen_grown_this_step * 1.0) + (power_produced_this_step * 0.1) + (strategic_reward * 0.1)
            reward = lichen_grown_this_step
            metrics["test_reward"] = test_reward

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        self.prev_obs = obs
        return obs

    def init_old_policy(self, obs_space, act_space, policy_kwargs):
        self.policy = MlpPolicy(
            obs_space,
            act_space,
            lr_schedule=lambda _: 3e-4,
            **policy_kwargs,
        )
    
    def set_old_policy(self, policy_state_dict):
        self.policy.load_state_dict(policy_state_dict)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent environment with a reduced observation and action space. It trains a policy that can succesfully control a heavy unit to dig ice and transfer it back to a factory to keep it alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=400,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model weights to use for evaluation"
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default="logs",
        help="Logging path",
    )
    args = parser.parse_args()
    return args


def make_env(env_id: str, type: str, rank: int, seed: int = 0, max_episode_steps=200):
    def _init() -> gym.Env:
        # verbose = 0
        # collect stats so we can create reward functions
        # max factories set to 4 for simplification and keeping returns consistent as we survive longer if there are more initial resources
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
        env.env_cfg.max_episode_steps = max_episode_steps

        # Add a SB3 wrapper to make it work with SB3 and simplify the action space with the controller
        # this will remove the bidding phase and factory placement phase. For factory placement we use
        # the provided place_near_random_ice function which will randomly select an ice tile and place a factory near it.
        env = SB3Wrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            bid_policy=zero_bid,
            # controller=(SimpleUnitDiscreteController(env.env_cfg) if type == "unit" else SimpleFactoryDiscreteController(env.env_cfg)),
        )

        env = (SimpleUnitObservationWrapper(env) if type == "unit" else SimpleFactoryObservationWrapper(env))
        env = (CustomUnitsEnvWrapper(env) if type == "unit" else CustomFactoriesEnvWrapper(env))
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)

        return env

    return _init

# class TensorboardCallback(BaseCallback):
#     def __init__(self, writer: SummaryWriter, tag: str = "", verbose: int = 0):
#         super().__init__(verbose)
#         self.writer = writer
#         self.tag = tag
#         self.global_step = 0  # 用来控制 tensorboard 的步数

#     def _on_step(self) -> bool:
#         for i, done in enumerate(self.locals["dones"]):
#             if done:
#                 info = self.locals["infos"][i]
#                 if "metrics" in info:
#                     for k, stat in info["metrics"].items():
#                         self.writer.add_scalar(f"{self.tag}/{k}", stat, self.global_step)
#         self.global_step += 1
#         return True

class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def save_model_state_dict(save_path, model):
    # save the policy state dict for kaggle competition submission
    state_dict = model.policy.to("cpu").state_dict()
    th.save(state_dict, save_path)


# def evaluate(args, env_id, model):
#     model = model.load(args.model_path)
#     video_length = 1000  # default horizon
#     eval_env = SubprocVecEnv(
#         [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
#     )
#     eval_env = VecVideoRecorder(
#         eval_env,
#         osp.join(args.log_path, "eval_videos"),
#         record_video_trigger=lambda x: x == 0,
#         video_length=video_length,
#         name_prefix=f"evaluation_video",
#     )
#     eval_env.reset()
#     out = evaluate_policy(model, eval_env, render=False, deterministic=False)
#     print(out)


# def train(args, env_id, units_model: PPO):
#     eval_env = SubprocVecEnv(
#         [make_env(env_id, "unit", i, max_episode_steps=1000) for i in range(4)]
#     )
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=osp.join(args.log_path, "models"),
#         log_path=osp.join(args.log_path, "eval_logs"),
#         eval_freq=24_000,
#         deterministic=False,
#         render=False,
#         n_eval_episodes=5,
#     )

#     units_model.learn(
#         args.total_timesteps,
#         callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
#     )
#     units_model.save(osp.join(args.log_path, "models/latest_model"))

def get_safe_state_dict(policy):
    import copy
    safe_sd = copy.deepcopy(policy.state_dict())
    for k, v in safe_sd.items():
        if isinstance(v, th.Tensor):
            safe_sd[k] = v.detach().cpu()
    return safe_sd

def train_units(args, env_id, units_model: PPO, factories_model: PPO, alternated_steps=10000, writer: SummaryWriter = None):
    eval_env = SubprocVecEnv(
        [make_env(env_id, "unit", i, max_episode_steps=1000) for i in range(4)]
    )
    eval_env.env_method(
        "init_old_policy",
        factories_model.observation_space,
        factories_model.action_space,
        factories_model.policy_kwargs,
    )
    eval_env.env_method(
        "set_old_policy",
        get_safe_state_dict(factories_model.policy),
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models/units"),
        log_path=osp.join(args.log_path, "eval_logs/units"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    units_model.get_env().env_method(
        "set_old_policy",
        get_safe_state_dict(factories_model.policy),
    )
    units_model.learn(
        alternated_steps,
        callback=[TensorboardCallback(tag="train_metrics/units"), eval_callback],
        reset_num_timesteps=False,
    )
    # units_model.save(osp.join(args.log_path, "models/units/latest_model"))

def train_factories(args, env_id, units_model: PPO, factories_model: PPO, alternated_steps=10000, writer: SummaryWriter = None):
    eval_env = SubprocVecEnv(
        [make_env(env_id, "factory", i, max_episode_steps=1000) for i in range(4)]
    )
    eval_env.env_method(
        "init_old_policy",
        units_model.observation_space,
        units_model.action_space,
        units_model.policy_kwargs,
    )
    eval_env.env_method(
        "set_old_policy",
        get_safe_state_dict(units_model.policy),
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models/factories"),
        log_path=osp.join(args.log_path, "eval_logs/factories"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    factories_model.get_env().env_method(
        "set_old_policy",
        get_safe_state_dict(units_model.policy),
    )
    factories_model.learn(
        alternated_steps,
        callback=[TensorboardCallback(tag="train_metrics/factories"), eval_callback],
        reset_num_timesteps=False,
    )
    # factories_model.save(osp.join(args.log_path, "models/factories/latest_model"))


def main(args):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    units_env = SubprocVecEnv(
        [
            make_env(env_id, "unit", i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    units_env.reset()
    factories_env = SubprocVecEnv(
        [
            make_env(env_id, "factory", i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    factories_env.reset()


    unit_obs_space = units_env.get_attr("observation_space", indices=0)[0]
    unit_act_space = units_env.get_attr("action_space", indices=0)[0]

    factory_obs_space = factories_env.get_attr("observation_space", indices=0)[0]
    factory_act_space = factories_env.get_attr("action_space", indices=0)[0]

    policy_kwargs = dict(net_arch=(128, 128))
    rollout_steps = 4096
    units_model = PPO(
        "MlpPolicy",
        units_env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=1024,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )
    factories_model = PPO(
        "MlpPolicy",
        factories_env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=1024,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )

    units_logger = configure(folder=osp.join(args.log_path, 'tb/units'), format_strings=["stdout", "tensorboard"])
    units_model.set_logger(units_logger)
    factories_logger = configure(folder=osp.join(args.log_path, 'tb/factories'), format_strings=["stdout", "tensorboard"])
    factories_model.set_logger(factories_logger)

    units_env.env_method(
        "init_old_policy",
        factory_obs_space,
        factory_act_space,
        policy_kwargs,
    )

    factories_env.env_method(
        "init_old_policy",
        unit_obs_space,
        unit_act_space,
        policy_kwargs,
    )

    units_steps = 400000
    factories_step = 200000
    # if args.eval:
    #     evaluate(args, env_id, model)
    # else:
    for iteration in range(args.total_timesteps // (units_steps + factories_step)):
        print(f"Training factories policy for {factories_step} steps")
        train_factories(args, env_id, units_model, factories_model, factories_step)
        gc.collect()
        print(f"Training units policy for {units_steps} steps")
        train_units(args, env_id, units_model, factories_model, units_steps)
        gc.collect()

    units_model.save(osp.join(args.log_path, "models/units/latest_model"))
    factories_model.save(osp.join(args.log_path, "models/factories/latest_model"))

if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())
