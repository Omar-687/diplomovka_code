import numpy as np
from scipy import stats
import math
import pytz
from datetime import datetime

import matplotlib.pyplot as plt

from acnportal import acnsim
from acnportal import algorithms
import gymnasium
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

# EV data management
# import gym_EV.envs.data_collection as data_collection# Get EV Charging Data
# import pymongo
# import bson
# from datetime import datetime, timedelta

# RL packages
import random  # Handling random number generation
from random import choices
from collections import deque  # Ordered collection with ends


class EVEnv(gymnasium.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, n_EVs=54, n_levels=10, max_capacity=20):
    # Parameter for reward function
    self.alpha = 0
    self.beta = 5
    self.gamma = 1

    # self.alpha = 1
    # self.beta = 1
    # self.gamma = 1
    self.signal = None
    self.state = None
    self.n_EVs = n_EVs
    self.n_levels = n_levels
    self._max_episode_steps = 100000
    self.flexibility = 0
    self.penalty = 0
    self.tracking_error = 0
    self.max_capacity = max_capacity
    self.max_rate = 6

    # Specify the observation space
    lower_bound = np.array([0])
    upper_bound = np.array([24, 70])
    low = np.append(np.tile(lower_bound, self.n_EVs * 2), lower_bound)
    high = np.append(np.tile(upper_bound, self.n_EVs), np.array([self.max_capacity]))
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Specify the action space
    upper_bound = self.max_rate
    low = np.append(np.tile(lower_bound, self.n_EVs), np.tile(lower_bound, self.n_levels))
    high = np.append(np.tile(upper_bound, self.n_EVs), np.tile(upper_bound, self.n_levels))
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Reset time for new episode
    # should be equal to arrival of first ev
    self.time = 0
    # each time step is period minutes
    self.time_interval = 1
    self.time_period = 5/60
    # store data
    self.data = None
  # TODO: this is bad step function, tongxin li used in aggregate flexibility branch the correct implementation
  def step(self, action):
    # Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval
    # Check if a new EV arrives
    for i in range(len(self.data)):
      if self.data[i][0] > self.time - self.time_interval and self.data[i][0] <= self.time:
        # Reject if all spots are full
        if np.where(self.state[:, 2] == 0)[0].size == 0:
          continue
        # Add a new active charging station
        else:
          self.state[np.where(self.state[:, 2] == 0)[0][0], 0] = self.data[i][1].ev.departure - self.data[i][1].ev.arrival
          self.state[np.where(self.state[:, 2] == 0)[0][0], 1] = self.data[i][1].ev.requested_energy
          self.state[np.where(self.state[:, 2] == 0)[0][0], 2] = 1

    # Update remaining time
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_result = self.state[:, 1] - action[:self.n_EVs] * self.time_period
    # Battery is full
    for item in range(len(charging_result)):
      if charging_result[item] < 0:
        action[item] = self.state[item, 1] / self.time_period
    self.state[:, 1] = charging_result.clip(min=0)

    self.penalty = 0
    for i in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time
      if self.state[i, 0] == 0:
        # The EV is overdue
        if self.state[i, 1] > 0:
          self.penalty = 10 * self.gamma * self.state[i, 1]
        # Deactivate the EV and reset
        self.state[i, :] = 0

      # Use soft penalty
      # else:
      #   penalty = self.gamma * self.state[0, 1] / self.state[i, 0]

    # Update rewards
    # Set entropy zero if feedback is allzero
    if not np.any(action[-self.n_levels:]):
      self.flexibility = 0
    else:
      self.flexibility = self.alpha * (stats.entropy(action[-self.n_levels:])) ** 2

    # if action gives cars more energy than the given signal, tracking error will be minus number
    # if action gives less than it will be plus
    self.tracking_error = self.beta * (np.sum(action[:self.n_EVs]) - self.signal) ** 2
    # by default self.flexibility will be 0 by these settings
    reward = (self.flexibility - self.tracking_error - self.penalty) / 100
    print('reward',reward)
    print('signal = ',self.signal)
    print('action = ', np.sum(action[:self.n_EVs]))
    # Select a new tracking signal
    levels = np.linspace(0, self.max_capacity, num=self.n_levels)
    # Set signal zero if feedback is allzero
    if not np.any(action[-self.n_levels:]):
      self.signal = 0
    else:
      self.signal = choices(levels, weights=action[-self.n_levels:])[0]

    done = True if self.time >= self.max_time else False
    terminated = done
    truncated = done
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    info = {}
    refined_act = action
    return obs, reward, terminated, truncated, info
  #  isTrain
  def reset(self, seed=None, options=None):
    # Select a random day and restart
    # TODO: get files for each day 1 day = 1 file
    isTrain = True

    # Timezone of the ACN we are using.
    timezone = pytz.timezone('America/Los_Angeles')

    start_day = random.randint(1, 60)
    # Start and End times are used when collecting data.
    start_month = 9 if start_day <= 30 else 10
    end_month = 9 if start_day <= 29 else 10
    start_day = start_day if start_day <= 30 else start_day - 30
    start = timezone.localize(datetime(2018, start_month, start_day))
    end = timezone.localize(datetime(2018, end_month, start_day + 1))

    # How long each time discrete time interval in the simulation should be.
    period = 5  # minutes

    # Voltage of the network.
    voltage = 220  # volts

    # Default maximum charging rate for each EV battery.
    default_battery_power = 32 * voltage / 1000  # kW

    # Identifier of the site where data will be gathered.
    site = 'caltech'
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)

    self.data = events.queue

    # if isTrain:
    #   day = random.randint(0, 99)
    #   name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_train/data' + str(day) + '.npy'
    # else:
    #   day = random.randint(0, 21)
    #   name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_test/data' + str(day) + '.npy'
    # Load data
    # data = np.load(name)
    # self.data = data
    first_ev = events.queue[0][1].ev
    # Initialize states and time
    self.state = np.zeros([self.n_EVs, 3])
    self.state[0, 0] = 0
    # Remaining time
    self.state[0, 0] = first_ev.departure - first_ev.arrival
    # # SOC
    self.state[0, 1] = first_ev.requested_energy
    # # The charging station is activated
    self.state[0, 2] = 1
    # Select initial signal to be zero -- does not matter since time interval is short
    self.signal = 0
    # self.time = np.floor(data[0, 0]*10) / 10.0
    self.time = first_ev.arrival

    last_ev = events.queue[-1][1].ev
    self.max_time =  last_ev.departure

    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    info = {}
    return obs, info