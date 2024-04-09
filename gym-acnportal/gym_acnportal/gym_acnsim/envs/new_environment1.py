#
#
# from copy import deepcopy
# from typing import Optional, Dict, List, Callable, Any
#
# import numpy as np
# from gymnasium import spaces
#
# from gym_acnportal.gym_acnsim.envs.base_env import BaseSimEnv
# import gym_acnportal.gym_acnsim.envs.observation  as obs
# import gym_acnportal.gym_acnsim.envs.reward_functions  as rf
# from gym_acnportal.gym_acnsim.envs.custom_envs import RebuildingEnv
# # import gym_acnportal.gym_acnsim.envs.observation as obs, reward_functions as rf
# # from gym_acnportal.gym_acnsim.envs.reward_functions import B
# from gym_acnportal.gym_acnsim.envs.action_spaces import SimAction, full_charging_schedule, zero_centered_single_charging_schedule
# from gym_acnportal.gym_acnsim.envs.observation import SimObservation
# from gym_acnportal.gym_acnsim.envs.interfaces import GymTrainedInterface
#
#
#
# #
# #
# #
# # # Default observation objects, action object, and reward functions list
# # # for use with make_default_sim_env and make_rebuilding_default_sim_env.
# default_observation_objects: List[SimObservation] = [
#     obs.arrival_observation(),
#     obs.departure_observation(),
#     obs.remaining_demand_observation(),
#     obs.constraint_matrix_observation(),
#     obs.magnitudes_observation(),
#     obs.phases_observation(),
#     obs.timestep_observation(),
# ]
# # default_action_object: SimAction = zero_centered_single_charging_schedule()
# # default_action_object: SimAction = centered_single_charging_schedule()
# default_action_object: SimAction = full_charging_schedule()
# default_reward_functions: List[Callable[[BaseSimEnv], float]] = [
#     rf.evse_violation,
#     rf.unplugged_ev_violation,
#     rf.current_constraint_violation,
#     rf.soft_charging_reward,
# ]
# #
# #
# # # TODO: check what i need to inherit to run my own environment
# # # TODO: change tab settings, tab should be equal to 3 spaces
# class EVEnv(BaseSimEnv):
#     observation_objects: List[SimObservation]
#     observation_space: spaces.Dict
#     action_object: SimAction
#     action_space: spaces.Space
#     reward_functions: List[Callable[[BaseSimEnv], float]]
#     def __init__(self,
#                interface: Optional[GymTrainedInterface],
#                observation_objects: List[SimObservation],
#                action_object: SimAction,
#                reward_functions: List[Callable[[BaseSimEnv], float]],
#                n_EVs=54,
#                n_levels=10,
#                max_capacity=20):
#     # Parameter for reward function
#         super().__init__(interface)
#         self.alpha = 0
#         self.beta = 5
#         self.gamma = 1
#         self.signal = None
#         self.state = None
#         self.n_EVs = n_EVs
#         self.n_levels = n_levels
#         self._max_episode_steps = 100000
#         self.flexibility = 0
#         self.penalty = 0
#         self.tracking_error = 0
#         self.max_capacity = max_capacity
#         self.max_rate = 6
#
#
#
#         self.observation_objects = observation_objects
#         self.action_object = action_object
#         self.reward_functions = reward_functions
#         if interface is None:
#           return
#         self.observation_space = spaces.Dict(
#           {
#             observation_object.name: observation_object.get_space(self.interface)
#             for observation_object in observation_objects
#           }
#         )
#
#     # self.action_space = action_object.get_space(interface)
#     # # Specify the observation space
#     # lower_bound = np.array([0])
#     # upper_bound = np.array([24, 70])
#     # low = np.append(np.tile(lower_bound, self.n_EVs * 2), lower_bound)
#     # high = np.append(np.tile(upper_bound, self.n_EVs), np.array([self.max_capacity]))
#     # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
#     #
#     # # Specify the action space
#     # upper_bound = self.max_rate
#     # low = np.append(np.tile(lower_bound, self.n_EVs), np.tile(lower_bound, self.n_levels))
#     # high = np.append(np.tile(upper_bound, self.n_EVs), np.tile(upper_bound, self.n_levels))
#     # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
#     #
#     # # Reset time for new episode
#     # self.time = 0
#     # self.time_interval = 0.1
#     # store data
#     # self.data = None
#
#     def action_to_schedule(self):
#         return self.action_object.get_schedule(self.interface, self.action)
#     def observation_from_state(self):
#         return {
#       observation_object.name: observation_object.get_obs(self.interface)
#       for observation_object in self.observation_objects
#     }
#
#     def reward_from_state(self):
#         return sum(
#       np.array([reward_func(self) for reward_func in self.reward_functions])
#     )
#     # def done_from_state(self):
#     #     return self.interface.is_done
#
#
#
#
#   # def step(self, action):
#   #   # Update states according to a naive battery model
#   #   # Time advances
#   #   self.time = self.time + self.time_interval
#   #   # Check if a new EV arrives
#   #   for i in range(len(self.data)):
#   #     if self.data[i, 0] > self.time - self.time_interval and self.data[i, 0] <= self.time:
#   #       # Reject if all spots are full
#   #       if np.where(self.state[:, 2] == 0)[0].size == 0:
#   #         continue
#   #       # Add a new active charging station
#   #       else:
#   #         self.state[np.where(self.state[:, 2] == 0)[0][0], 0] = self.data[i, 1]
#   #         self.state[np.where(self.state[:, 2] == 0)[0][0], 1] = self.data[i, 2]
#   #         self.state[np.where(self.state[:, 2] == 0)[0][0], 2] = 1
#   #
#   #   # Update remaining time
#   #   time_result = self.state[:, 0] - self.time_interval
#   #   self.state[:, 0] = time_result.clip(min=0)
#   #
#   #   # Update battery
#   #   charging_result = self.state[:, 1] - action[:self.n_EVs] * self.time_interval
#   #   # Battery is full
#   #   for item in range(len(charging_result)):
#   #     if charging_result[item] < 0:
#   #       action[item] = self.state[item, 1] / self.time_interval
#   #   self.state[:, 1] = charging_result.clip(min=0)
#   #
#   #   self.penalty = 0
#   #   for i in np.nonzero(self.state[:, 2])[0]:
#   #     # The EV has no remaining time
#   #     if self.state[i, 0] == 0:
#   #       # The EV is overdue
#   #       if self.state[i, 1] > 0:
#   #         self.penalty = 10 * self.gamma * self.state[i, 1]
#   #       # Deactivate the EV and reset
#   #       self.state[i, :] = 0
#   #
#   #     # Use soft penalty
#   #     # else:
#   #     #   penalty = self.gamma * self.state[0, 1] / self.state[i, 0]
#   #
#   #   # Update rewards
#   #   # Set entropy zero if feedback is allzero
#   #   if not np.any(action[-self.n_levels:]):
#   #     self.flexibility = 0
#   #   else:
#   #     self.flexibility = self.alpha * (stats.entropy(action[-self.n_levels:])) ** 2
#   #
#   #   self.tracking_error = self.beta * (np.sum(action[:self.n_EVs]) - self.signal) ** 2
#   #   reward = (self.flexibility - self.tracking_error - self.penalty) / 100
#   #
#   #   # Select a new tracking signal
#   #   levels = np.linspace(0, self.max_capacity, num=self.n_levels)
#   #   # Set signal zero if feedback is allzero
#   #   if not np.any(action[-self.n_levels:]):
#   #     self.signal = 0
#   #   else:
#   #     self.signal = choices(levels, weights=action[-self.n_levels:])[0]
#   #
#   #   done = True if self.time >= 24 else False
#   #   obs = np.append(self.state[:, 0:2].flatten(), self.signal)
#   #   info = {}
#   #   refined_act = action
#   #   return obs, reward, done, info, refined_act
#   #
#   #
#   #
#   # # TODO: change input parameters and the way it is loaded
#   # def reset(self, isTrain):
#   #   # Select a random day and restart
#   #   if isTrain:
#   #     day = random.randint(0, 99)
#   #     name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_train/data' + str(day) + '.npy'
#   #   else:
#   #     day = random.randint(0, 21)
#   #     name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_test/data' + str(day) + '.npy'
#   #   # Load data
#   #   data = np.load(name)
#   #   self.data = data
#   #   # Initialize states and time
#   #   self.state = np.zeros([self.n_EVs, 3])
#   #   # Remaining time
#   #   self.state[0, 0] = data[0, 1]
#   #   # SOC
#   #   self.state[0, 1] = data[0, 2]
#   #   # The charging station is activated
#   #   self.state[0, 2] = 1
#   #   # Select initial signal to be zero -- does not matter since time interval is short
#   #   self.signal = 0
#   #   # self.time = np.floor(data[0, 0]*10) / 10.0
#   #   self.time = data[0, 0]
#   #
#   #   obs = np.append(self.state[:, 0:2].flatten(), self.signal)
#   #   return obs
# def make_ev_sim_env(
#             interface: Optional[GymTrainedInterface] = None,
#     ):
#       return EVEnv(interface=interface,
#                    observation_objects=default_observation_objects,
#                    action_object=default_action_object,
#                    reward_functions=default_reward_functions)
#
# def make_rebuilding_ev_sim_env(
#           interface_generating_function: Optional[Callable[[], GymTrainedInterface]]
#   ) -> RebuildingEnv:
#     """ A simulator environment with the same characteristics as the
#     environment returned by make_default_sim_env except on every reset,
#     the simulation is completely rebuilt using interface_generating_function.
#
#     See make_default_sim_env for more info.
#     """
#     interface = interface_generating_function()
#     return RebuildingEnv.from_custom_sim_env(
#       make_ev_sim_env(interface),
#       interface_generating_function=interface_generating_function,
#     )