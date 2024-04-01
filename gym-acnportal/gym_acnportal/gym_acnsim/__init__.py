# # coding=utf-8
# """
# Open AI Gym plugin for ACN-Sim. Provides several customizable
# environments for training reinforcement learning (RL) agents. See
# tutorial X for examples of usage.
# """
# # rather use gymnasium name instead of gym to make things clear
# from typing import List, Dict
# import gymnasium
# # from gymnasium.envs import registry
# from gymnasium.envs.registration import registry, register, EnvSpec
# # from envs import CustomSimEnv
# # from .interfaces import GymTrainedInterface, GymTrainingInterface
# # from .envs import *
# # from gym_acnportal.gym_acnsim.interfaces import  GymTrainedInterface, GymTrainingInterface
# # from gym_acnportal.gym_acnsim.envs import *
# all_envs: List[EnvSpec] = list(registry.values())
# env_ids = [env_spec.id for env_spec in all_envs]
# gym_env_dict: Dict[str, str] = {
#     "custom-acnsim-v0": "gym_acnportal.gym_acnsim.envs:CustomSimEnv",
#     "default-acnsim-v0": "gym_acnportal.gym_acnsim.envs:make_default_sim_env",
#     "rebuilding-acnsim-v0": "gym_acnportal.gym_acnsim.envs:RebuildingEnv",
#     "default-rebuilding-acnsim-v0": "gym_acnportal.gym_acnsim.envs:make_rebuilding_default_sim_env",
# }
#
# # gym_env_dict: Dict[str, str] = {
# #     "custom-acnsim-v0": CustomSimEnv,
# #     "default-acnsim-v0": "envs:make_default_sim_env",
# #     "rebuilding-acnsim-v0": "envs:RebuildingEnv",
# #     "default-rebuilding-acnsim-v0": "envs:make_rebuilding_default_sim_env",
# # }
#
# # registration test
# for env_name, env_entry_point in gym_env_dict.items():
#     if env_name not in env_ids:
#         register(id=env_name, entry_point=env_entry_point)
# env_name = "default-rebuilding-acnsim-v0"
# for key in gym_env_dict.keys():
#     try:
#         gymnasium.make(key)
#         print(f"The '{key}' environment exists.")
#     except gymnasium.error.UnregisteredEnv:
#         print(f"The '{key}' environment does not exist.")
#     except Exception as e:
#         # Handling other types of exceptions
#         print("Environment registered. An error occurred:", e)
# del register, registry, all_envs, gym_env_dict, List, Dict
