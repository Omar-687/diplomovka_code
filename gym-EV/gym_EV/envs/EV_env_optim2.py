import numpy as np
from scipy import stats
import math
import pytz
from datetime import datetime
from gym_EV.envs.utils import (plot_daily_price,
                               plot_substation_rates_given_by_operator,
                               plot_charging_rate_over_time,
                               plot_table_for_other_info)

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

    # self.xi = 2
    self.xi = 1
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
    self.charged_energy_over_day = []
    self.initial_bat = []
    self.dic_bat = {}
    self.day = None
    # self.selection = "Random"
    self.selection = "Optimization"
    self.price = []
    self.start_date = None
    self.end_date = None
    self.cost = 0
    self.power = 0
    self.tuning_parameter = tuning
    self.total_flexibility = 0
    self.total_charging_error = 0
    self.total_tracking_error = 0
    self.total_reward = 0
    self.all_requested_energy = 0

    # Specify the observation space
    lower_bound = np.array([0])
    # max requested energy set to 70 kWh
    # TODO: if found any of these assumptions false, change upper bound or data about EVs
    upper_bound = np.array([24, 70])
    low = np.append(np.tile(lower_bound, self.max_ev * 2), lower_bound)
    high = np.append(np.tile(upper_bound, self.max_ev), np.array([self.max_capacity]))
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    self.overall_cost = 0
    self.overall_energy_delivered = 0
    self.overall_requested_energy = 0

    # TODO:how to change env variable before evaluating
    # self.evaluation = False
    # self.evaluation = True
    self.evaluation = False

    self.generated_signals_in_one_day = []
    self.number_of_evs = 0
    mpe_undelivered_energy_error = 0
    self.done = False

    # Specify the action space
    # upper_bound = self.max_rate
    upper_bound = 1
    low = np.tile(lower_bound, self.number_level)
    high = np.tile(upper_bound, self.number_level)
    # test with action space [-1,1] and also rescaling or find if it is ok to have [0,1]
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    self.time_step = 1
    self.time_normalised = 0
    # Reset time for new episode
    # Time unit is measured in Hr
    self.time = 0
    self.time_period = 12
    self.time_interval = self.time_period/60
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
    # update global time
    self.time = self.time + self.time_interval
    self.time_normalised = self.time_normalised + self.time_step
    self.mark = self.mark + 1

    # changed price according article
    self.price = np.append(self.price, 1 - self.time / 24)

    # EDF sorting according arrival from first to last
    self.state = np.array(sorted(self.state,key=lambda x: x[0]))
    tmp_signal = 0
    if self.mark == self.control_steps:
      self.mark = 0

      if self.selection == "Optimization":
        tmp_signal = self.select_signal(flexibility_feedback=action, current_price=self.price[-1])

      self.signal = self.smoothing * np.mean(self.signal_buffer) + (1 - self.smoothing) * tmp_signal
      self.signal_buffer.append(self.signal)
      self.generated_signals_in_one_day.append(self.signal)
    # Time advances
    self.all_requested_energy = 0

    # Check if a new EV arrives
    for i in range(len(self.data)):
      self.all_requested_energy += self.data[i][1].ev.requested_energy
      if self.data[i][0] > self.time_normalised - self.time_step and self.data[i][0] <= self.time_normalised:
        # Reject if all spots are full
        if np.where(self.state[:, 2] == 0)[0].size == 0:
          continue
        # Add a new active charging station
        else:
          idx = np.where(self.state[:, 2] == 0)[0][0]
          self.state[idx, 0] = (self.data[i][1].ev.departure - self.data[i][1].ev.arrival)*(self.time_interval)
          self.state[idx, 1] =  self.data[i][1].ev.requested_energy
          self.state[idx, 2] = 1

    # process of charging
    self.remaining_signal = self.signal
    given_charging_rates = []
    for i in range(len(self.state)):
      evse = self.state[i]
      if evse[0] == 0:
        continue
      remaining_demand = evse[1]
      charging_rate = min(remaining_demand,
                          self.remaining_signal,
                          self.max_rate) * self.time_interval
      given_charging_rates.append(charging_rate)
      self.remaining_signal -= charging_rate
      self.state[i][1] -= charging_rate

    # Update remaining time
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)


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


    # Update rewards
    undelivered_energy = abs(self.signal - self.remaining_signal)
    # Compute costs
    self.power = np.sum(self.signal - undelivered_energy)
    temp_cost = self.xi * self.power * self.price[-1]
    self.cost = self.cost + temp_cost
    self.overall_cost += temp_cost

    # Set entropy zero if feedback is allzero
    if not np.any(action):
      self.flexibility = 0
    else:
      self.flexibility = self.alpha * stats.entropy(action)
      self.total_flexibility = self.total_flexibility + self.flexibility

    # Compute tracking error
    self.tracking_error = self.beta * abs(np.sum(given_charging_rates) - self.signal)
    self.charged_energy_over_day.append(np.sum(given_charging_rates))

    # o1, o2, o3 = 0.1, 0.2, 2
    o1, o2, o3 = 0.1, 10, 2
    reward = (self.flexibility +
              o1 * np.linalg.norm(given_charging_rates) -
              o2 * self.penalty -
              o3 * self.tracking_error)
    self.total_reward += reward


    # end episode at the end of the given day
    done = True if self.time >= 24 or self.done else False
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    # TODO: change observation space
    # obs = self.state
    info = {'action': action}
    # termination and truncation is same in our case, for better charging visualisation
    terminated = done
    truncated = done
    return obs, reward, terminated, truncated, info

  def generate_random_datetime(self,start_year, start_month, start_day, end_year, end_month, end_day):
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day)

    delta = end_date - start_date
    random_number_of_days = random.randint(0, delta.days)

    random_date = start_date + timedelta(days=random_number_of_days)

    return random_date

  def reset(self, seed=None, options=None):
    # self.evaluation and
    if len(self.generated_signals_in_one_day) != 0:
      start_date_str =  self.start_date.strftime("%d.%m.%Y")
      end_date_str = self.end_date.strftime("%d.%m.%Y")
      # plot_substation_rates_given_by_operator(self.generated_signals_in_one_day,
      #                                         self.time_period,
      #                                         start_date_str,
      #                                         end_date_str)
      # plot_charging_rate_over_time(self.charged_energy_over_day,
      #                              self.time_period,
      #                              start_date_str,
      #                              end_date_str)
      overall_energy_charged = np.sum(self.charged_energy_over_day)
      number_of_evs = len(self.data)
      plot_table_for_other_info(overall_energy_charged=overall_energy_charged,
                                overall_energy_requested=self.all_requested_energy,
                                total_charging_costs=self.overall_cost,
                                number_of_evs=number_of_evs,
                                start_date=start_date_str,
                                end_date=end_date_str)

    # Timezone of the ACN we are using.
    timezone = pytz.timezone('America/Los_Angeles')

    # random date from 1st Nov 2018-1st Dec 2019
    random_date = self.generate_random_datetime(2018,11,1,
                                                2019,12,1)
    random_end_date = random_date + timedelta(days=1)
    start = timezone.localize(random_date)
    self.start_date = start
    end = timezone.localize(random_end_date)
    self.end_date = end

    # How long each time discrete time interval in the simulation should be.
    period = self.time_period  # minutes

    # Voltage of the network.
    voltage = 220  # volts

    # Default maximum charging rate for each EV battery.
    default_battery_power = 32 * voltage / 1000  # kW

    # Identifier of the site where data will be gathered.
    site = 'caltech'
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)

    self.data = events.queue
    # change done to be global
    self.done = 0
    if len(self.data) == 0:
      self.done = 1

    # # Reset values
    self.flexibility = 0
    self.tracking_error = 0

    # Initialize states and time
    self.state = np.zeros([self.max_ev, 3])
    self.time = 0
    self.time_normalised = 0

    # Select initial signal as 0
    self.signal = 0
    self.power = 0
    # signal buffer
    self.signal_buffer.clear()
    self.signal_buffer.append(self.signal)
    self.charging_result = []
    self.charged_energy_over_day = []

    # Generate random price sequence
    # self.price = np.append(self.price, random.random())
    self.generated_signals_in_one_day = []
    self.price = []
    self.overall_cost = 0
    self.cost = 0
    self.total_flexibility = 0
    self.total_charging_error = 0
    self.total_tracking_error = 0
    self.total_reward = 0
    self.penalty = 0
    obs = np.append(self.state[:, 0:2].flatten(), self.signal)
    return obs, {}
