import numpy as np
from scipy import stats
import math
import pytz
from datetime import datetime


from decimal import Decimal
import gymnasium
from gymnasium import error, spaces, utils
from gym.utils import seeding
from acnportal import acnsim
from acnportal import algorithms
# EV data management
# import gym_EV.envs.data_collection as data_collection# Get EV Charging Data
# import pymongo
# import bson
# from datetime import datetime, timedelta

# RL packages
import random  # Handling random number generation
from random import choices
from collections import deque  # Ordered collection with ends
from datetime import datetime, timedelta
""""
Version: AG-v0.0
We define a simple EV charging environment for aggregate flexibility characterization. 
The power signal is sampled according to the feedback and the operational constraints (peak power limit).

Update: AG-v0.1
Adding a component for online cost minimization
"""

class EVEnvOptim(gymnasium.Env):
  metadata = {'render.modes': ['human']}

  # tuning = 50 before, according to article it should be 6 * 10^3
  # max capacity should be 150, not 20
  def __init__(self, max_ev=54, number_level=10, max_capacity=20,  max_rate=6.6, tuning=6e3):
    # Parameter for reward function
    self.alpha = 1
    self.beta = 1
    self.gamma = 1
    self.xi = 2
    # TODO: try with action space [0,1]^10, not actions
    self.data = None
    self.signal = None
    self.state = None
    self.peak_power = 50
    self.max_ev = max_ev
    self.number_level = number_level
    self._max_episode_steps = 100000
    self.flexibility = 0
    self.total_flexibility = 0
    self.penalty = 0
    self.tracking_error = 0
    self.max_capacity = max_capacity
    self.max_rate = max_rate
    # store previous signal for smoothing
    self.signal_buffer = deque(maxlen=5)
    self.smoothing = 0.0
    # store EV charging result
    self.charging_result = []
    self.initial_bat = []
    self.dic_bat = {}
    self.day = None
    # self.selection = "Random"
    self.selection = "Optimization"
    self.price = []
    self.cost = 0
    self.power = 0
    self.tuning_parameter = tuning
    self.total_flexibility = 0
    self.total_charging_error = 0
    self.total_tracking_error = 0
    self.total_reward = 0
    self.map_evse_to_ev = {}
    # Specify the observation space
    lower_bound = np.array([0])
    upper_bound = np.array([24, 70])
    low = np.append(np.tile(lower_bound, self.max_ev * 2), lower_bound)
    high = np.append(np.tile(upper_bound, self.max_ev), np.array([self.max_capacity]))
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Specify the action space
    upper_bound = self.max_rate
    low = np.append(np.tile(lower_bound, self.max_ev), np.tile(lower_bound, self.number_level))
    high = np.append(np.tile(upper_bound, self.max_ev), np.tile(upper_bound, self.number_level))
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    self.time_step = 1
    self.time_normalised = 0
    # Reset time for new episode
    # Time unit is measured in Hr
    self.time = 0
    self.time_interval = 12/60
    # This decides the granuality of control
    self.control_steps = 1
    self.mark = 0
    # store data
    self.data = None

  def select_signal(self, flexibility_feedback, current_price):

    signal = 0
    levels = np.linspace(0, self.max_capacity, num=self.number_level)
    if not np.any(flexibility_feedback):
      flexibility_feedback[0] = 1
    flexibility_feedback = flexibility_feedback / sum(flexibility_feedback)
    objective_temp = 1000000
    # max_objective = -np.inf
    for label in range(len(flexibility_feedback)):
      if flexibility_feedback[label] > 0.01:
        objective = current_price * levels[label] - self.tuning_parameter * math.log(flexibility_feedback[label])

        # objective = current_price * levels[label]
        if objective < objective_temp:
          signal = levels[label]
          objective_temp = objective
    # print('objective: ',objective)
    # print('log prob ',math.log(flexibility_feedback[label]))
    return signal

  def step(self, action):

    # Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval
    self.time_normalised = self.time_normalised + self.time_step
    self.mark = self.mark +1
    # changed price according article
    # Price updates
    self.price = np.append(self.price, 1 - self.time / 24)
    # Check if a new EV arrives
    for i in range(len(self.data)):
      if self.data[i][0] > self.time_normalised - self.time_step and self.data[i][0] <= self.time_normalised:
        # Reject if all spots are full
        if np.where(self.state[:, 2] == 0)[0].size == 0:
          continue
        # Add a new active charging station
        else:
          idx = np.where(self.state[:, 2] == 0)[0][0]
          self.state[idx, 0] = self.data[i][1].ev.departure - self.data[i][1].ev.arrival
          self.state[idx, 1] =  self.data[i][1].ev.requested_energy
          self.state[idx, 2] = 1
          self.map_evse_to_ev[idx] = self.data[i][1].ev
          # self.dic_bat[idx] = self.data[i, 2]

    # Project action
    action[np.where(self.state[:,2] == 0)[0]] = 0
    if np.sum(action[:self.max_ev]) > self.max_capacity:
      action[:self.max_ev] = 1.0 * action[:self.max_ev] * self.max_capacity / np.sum(action[:self.max_ev])

    # Update remaining time
    # time_result = self.state[:, 0] - self.time_interval
    time_result = self.state[:, 0] - self.time_step
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_state = self.state[:, 1] - action[:self.max_ev] * self.time_interval

    # Battery is full
    for item in range(len(charging_state)):
      if charging_state[item] < 0:
        action[item] = self.state[item, 1] / self.time_interval
    self.state[:, 1] = charging_state.clip(min=0)



    self.penalty = 0
    for i in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time
      if self.state[i, 0] == 0:
        # The EV is overdue
        self.charging_result = np.append(self.charging_result, self.state[i,1])
        # self.initial_bat.append(self.dic_bat[i])
        if self.state[i, 1] > 0:
          self.penalty = self.gamma * self.state[i, 1]
        # Deactivate the EV and reset
        self.state[i, :] = 0
        self.map_evse_to_ev[i] = None

      # Use soft penalty
      # else:
      #   penalty = self.gamma * self.state[0, 1] / self.state[i, 0]

    # Select a new tracking signal

    # Operational constraints: Peak Power Bound (PPB):
    action_controlling = action[self.max_ev:]

    # for i in range(len(action_controlling)):
    #   if i > self.peak_power:
    #     action_controlling[i] = 0
    tmp_signal = 0
    if self.mark == self.control_steps:
      self.mark = 0

      if self.selection == "Optimization":

        tmp_signal = self.select_signal(action_controlling, self.price[-1])

      # if self.selection == "Random":
      #
      #   levels = np.linspace(0, self.max_capacity, num=self.number_level)
      #
      #   # Set signal zero if feedback is all-zero
      #   if not np.any(action):
      #     action_controlling[0] = 1
      #     tmp_signal = 0
      #   else:
      #     # normalize flexibility feedback
      #     action = action / np.sum(action[-self.number_level:])
      #     # Soft-selection
      #     tmp_signal = choices(levels, weights=action_controlling)[0]
          # Hard-selection
          # tmp_signal = levels[np.argmax(action_controlling)]

      self.signal = self.smoothing * np.mean(self.signal_buffer) + (1-self.smoothing) * tmp_signal
      self.signal_buffer.append(self.signal)

    # Update rewards

    # Compute costs
    self.power = np.sum(action[:self.max_ev])
    temp_cost = self.xi * self.power * self.price[-1]
    self.cost = self.cost + temp_cost

    # Set entropy zero if feedback is allzero
    if not np.any(action_controlling):
      self.flexibility = 0
    else:
      self.flexibility = self.alpha * stats.entropy(action_controlling)
      self.total_flexibility = self.total_flexibility + self.flexibility
    # Compute tracking error
    self.tracking_error = self.beta * abs(np.sum(action[:self.max_ev]) - self.signal)
    o1,o2,o3= 0.1, 0.2, 2
    # reward = (self.flexibility - self.tracking_error - self.penalty)
    # reward = (self.flexibility - self.tracking_error - self.penalty) / 100
    # reward = (self.flexibility - self.tracking_error - self.penalty) / 100
    charging_performance = 0
    for idx,value in self.map_evse_to_ev.items():
      if value == None:
        continue
      ev = value
      charging_performance += ev.requested_energy - self.state[idx,1]
    #
    reward = (self.flexibility + o1*np.linalg.norm(action[:self.max_ev]) - o2*self.penalty - o3*self.tracking_error) / 100
    self.total_reward += reward
    # print()
    # print('reward', reward, f'flexibility = {self.flexibility}')
    # print('second term :', o1*np.linalg.norm(action[:self.max_ev]))
    # print('charging performance:', o2*self.penalty)
    # print('tracking performance', o3*self.tracking_error)
    # print()


    # print('charging performance', 0.2*charging_performance)
    # print('tracking error : ', 2*self.tracking_error)
    # print('signal = ', self.signal)
    # print('action = ', np.sum(action[:self.max_ev]))

    # Return obs and rewards
    done = True if self.time >= 24 else False
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    info = {}
    terminated = done
    truncated = done
    refined_act = action
    return obs, reward, terminated, truncated, info

  # def sample_episode(self, isTrain):
  #   # Sample depends on Train/Test
  #   if isTrain:
  #     self.day = random.randint(0, 99)
  #     name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_train/data' + str(self.day) + '.npy'
  #   else:
  #     self.day = random.randint(0, 21)
  #     name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_test/data' + str(self.day) + '.npy'
  #     # Load data
  #   data = np.load(name)
  #   return self.day, data

  def get_episode_by_time(self, day):
    self.day = day
    # name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_one/data' + str(self.day) + '.npy'
    # name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_greedy/data' + str(self.day) + '.npy'
    # name = '/Users/tonytiny/Documents/Github/gym-EV_data/fake/data' + str(self.day) + '.npy'
    # name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_greedy_jpl/data' + str(self.day) + '.npy'
    name = '/Users/tonytiny/Documents/Github/gym-EV_data/selected_day/data' + str(self.day) + '.npy'
    # Load data
    data = np.load(name)
    return self.day, data

  def generate_random_datetime(self,start_year, start_month, start_day, end_year, end_month, end_day):
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day)

    delta = end_date - start_date
    random_number_of_days = random.randint(0, delta.days)

    random_date = start_date + timedelta(days=random_number_of_days)

    return random_date
  # , day
  def reset(self, seed=None, options=None):
    # Select a random day and restart
    # _, self.data = self.sample_episode(isTrain)

    # Select day in a chronological order
    # _, self.data = self.get_episode_by_time(day)

    # Timezone of the ACN we are using.
    timezone = pytz.timezone('America/Los_Angeles')

    # start_day = random.randint(1, 60)
    # # Start and End times are used when collecting data.
    # start_month = 9 if start_day <= 30 else 10
    # end_month = 9 if start_day <= 29 else 10
    # start_day = start_day if start_day <= 30 else start_day - 30
    random_date = self.generate_random_datetime(2018,11,1,2019,12,1)
    random_end_date = random_date + timedelta(days=1)
    start = timezone.localize(random_date)
    end = timezone.localize(random_end_date)

    # How long each time discrete time interval in the simulation should be.
    period = 12  # minutes

    # Voltage of the network.
    voltage = 220  # volts

    # Default maximum charging rate for each EV battery.
    default_battery_power = 32 * voltage / 1000  # kW

    # Identifier of the site where data will be gathered.
    site = 'caltech'
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)

    self.data = events.queue
    done = 0
    if len(self.data) == 0:
      done = 1
      obs = []
      return obs, done

    # Reset values
    self.signal = None
    self.state = None
    self.flexibility = 0
    self.total_flexibility = 0
    self.penalty = 0
    self.tracking_error = 0
    self.charging_result = []
    # self.initial_bat = []
    # self.dic_bat = {}
    # self.day = day

    # Initialize states and time

    self.state = np.zeros([self.max_ev, 3])
    # # Remaining time
    # self.state[0, 0] = self.data[0, 1]
    # # SOC
    # self.state[0, 1] = self.data[0, 2]
    # # The charging station is activated
    # self.state[0, 2] = 1
    # print('total reward : ',self.total_reward)
    # Start from 0 for better visualization
    # self.time = self.data[0, 0]
    self.time = 0
    self.time_normalised = 0
    # Select initial signal randomly -- does not matter since time interval is short
    # levels = np.linspace(0, self.max_capacity, num=self.number_level)
    # self.signal = choices(levels)[0]
    # Select initial signal as 0
    self.signal = 0
    self.power = 0
    # signal buffer
    self.signal_buffer.clear()
    self.signal_buffer.append(self.signal)
    self.charging_result = []
    # self.initial_bat = []
    # self.dic_bat = {}
    # self.dic_bat[0] = self.data[0, 2]

    # Generate random price sequence
    # self.price = np.append(self.price, random.random())
    self.price = []
    self.cost = 0
    self.total_flexibility = 0
    self.total_charging_error = 0
    self.total_tracking_error = 0
    self.total_reward = 0

    self.penalty = 0
    self.charging_result = []
    # self.initial_bat = []
    # self.dic_bat[0] = self.data[0, 2]
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    return obs, {'done':done}