import numpy as np
from acnportal.algorithms import BaseAlgorithm
import copy
import random
import cvxpy as cp
import numpy
from acnportal.acnsim.events import PluginEvent
from acnportal.acnsim.models import Battery
from acnportal.acnsim.models import EV
from utils import plot_charging_profiles
from acnportal.algorithms.preprocessing import *
class SmoothedLLF(BaseAlgorithm):
    def __init__(self, increment=1):
        super().__init__()
        self._increment = increment
        self.max_recompute = 1


    def get_laxity(self,ev:SessionInfo):
        return (ev.estimated_departure - self.interface.current_time) - (self.interface.remaining_amp_periods(ev) / self.interface.max_pilot_signal(ev.station_id))

    def schedule(self, active_evs):
        if len(active_evs) == 0:
            return {}


        schedule = {ev.station_id: [0] for ev in active_evs}
        # self.infrastructure: InfrastructureInfo = self.interface.get_infrastructure()
        # active_evs = sorted(active_evs, key=self.get_laxity)
        # TODO: how to get P(t) time varying capacity
        # TODO: find how to get P(t) in acn package, try to find it in large scale adaptive
        P = 100
        # TODO: find a reference to P_min and P_max, how they were calculated


        minimum_laxity_Lt_lower_bound = np.inf
        maximum_laxity_Lt_upper_bound = -np.inf

        for ev in active_evs:
            ev_laxity = self.get_laxity(ev)

            if ev_laxity - 1 < minimum_laxity_Lt_lower_bound:
                minimum_laxity_Lt_lower_bound = ev_laxity - 1

            if ev_laxity > maximum_laxity_Lt_upper_bound:
                maximum_laxity_Lt_upper_bound = ev_laxity


        error_tolerance = 1e-4
        max_iterations = 600
        iteration = 0
        update_schedule = {}
        L_t = (minimum_laxity_Lt_lower_bound + maximum_laxity_Lt_upper_bound) / 2
        while (iteration <= max_iterations
               and abs(maximum_laxity_Lt_upper_bound - minimum_laxity_Lt_lower_bound) > error_tolerance):


            for ev in active_evs:
                peak_charging_rate = self.interface.max_pilot_signal(ev.station_id)

                schedule[ev.station_id] = list(np.clip([peak_charging_rate * (L_t - self.get_laxity(ev) + 1)],
                                                         0,
                                                         min(peak_charging_rate, self.interface.remaining_amp_periods(ev))))
            if not self.interface.is_feasible(schedule):
                maximum_laxity_Lt_upper_bound = L_t
            else:
                minimum_laxity_Lt_lower_bound = L_t

            L_t = (minimum_laxity_Lt_lower_bound + maximum_laxity_Lt_upper_bound) / 2
            iteration += 1
        if not self.interface.is_feasible(schedule):
            for ev in active_evs:
                peak_charging_rate = self.interface.max_pilot_signal(ev.station_id)

                schedule[ev.station_id] = list(np.clip([peak_charging_rate * (minimum_laxity_Lt_lower_bound - self.get_laxity(ev) + 1)],
                                                         0,
                                                         min(peak_charging_rate, self.interface.remaining_amp_periods(ev))))

        # for ev in active_evs:
        #     schedule[ev.station_id] = ev.max_rates[0] * (L_t - self.get_laxity(ev) + 1)

        return schedule

            # TODO: check uncontrolled charging if it is possible to turn off the infrastructure checks
            # TODO: find reference to r_min and r_max and how they were calculated
            # peak charging rate r is given
            # r = self.interface.max_pilot_signal(ev.station_id)
            # r_min, r_max = 0, r*2





        #
        #
        # return schedule

"""
ACN-Sim Tutorial: Lesson 2
Developing a Custom Algorithm
by Zachary Lee
Last updated: 03/19/2019
--

In this lesson we will learn how to develop a custom algorithm and run it using ACN-Sim. For this example we will be
writing an Earliest Deadline First Algorithm. This algorithm is already available as part of the SortingAlgorithm in the
algorithms package, so we will compare the results of our implementation with the included one.
"""

# -- Custom Algorithm --------------------------------------------------------------------------------------------------
from acnportal.algorithms import BaseAlgorithm

# All custom algorithms should inherit from the abstract class BaseAlgorithm. It is the responsibility of all derived
# classes to implement the schedule method. This method takes as an input a list of EVs which are currently connected
# to the system but have not yet finished charging. Its output is a dictionary which maps a station_id to a list of
# charging rates. Each charging rate is valid for one period measured relative to the current period.
# For Example:
#   * schedule['abc'][0] is the charging rate for station 'abc' during the current period
#   * schedule['abc'][1] is the charging rate for the next period
#   * and so on.
#
# If an algorithm only produces charging rates for the current time period, the length of each list should be 1.
# If this is the case, make sure to also set the maximum resolve period to be 1 period so that the algorithm will be
# called each period. An alternative is to repeat the charging rate a number of times equal to the max recompute period.


class EarliestDeadlineFirstAlgo(BaseAlgorithm):
    """ Algorithm which assigns charging rates to each EV in order or departure time.

    Implements abstract class BaseAlgorithm.

    For this algorithm EVs will first be sorted by departure time. We will then allocate as much current as possible
    to each EV in order until the EV is finished charging or an infrastructure limit is met.

    Args:
        increment (number): Minimum increment of charging rate. Default: 1.
    """

    def __init__(self, increment=1):
        super().__init__()
        self._increment = increment
        self.max_recompute = 1

    def schedule(self, active_evs):
        """ Schedule EVs by first sorting them by departure time, then allocating them their maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        # First we define a schedule, this will be the output of our function
        schedule = {ev.station_id: [0] for ev in active_evs}

        # Next, we sort the active_evs by their estimated departure time.
        sorted_evs = sorted(active_evs, key=lambda x: x.estimated_departure)

        # We now iterate over the sorted list of EVs.
        for ev in sorted_evs:
            # First try to charge the EV at its maximum rate. Remember that each schedule value must be a list, even
            #   if it only has one element.
            schedule[ev.station_id] = [self.interface.max_pilot_signal(ev.station_id)]

            # If this is not feasible, we will reduce the rate.
            #   interface.is_feasible() is one way to interact with the constraint set of the network. We will explore
            #   another more direct method in lesson 3.
            while not self.interface.is_feasible(schedule):
                # Since the maximum rate was not feasible, we should try a lower rate.
                schedule[ev.station_id][0] -= self._increment

                # EVs should never charge below 0 (i.e. discharge) so we will clip the value at 0.
                if schedule[ev.station_id][0] < 0:
                    schedule[ev.station_id] = [0]
                    break
        return schedule


# -- Run Simulation ----------------------------------------------------------------------------------------------------
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy

from acnportal import acnsim
from acnportal import algorithms

# Now that we have implemented our algorithm, we can try it out using the same experiment setup as in lesson 1.
# The only difference will be which scheduling algorithm we use.
from acnportal.signals.tariffs.tou_tariff import TimeOfUseTariff



# -- Experiment Parameters ---------------------------------------------------------------------------------------------
timezone = pytz.timezone("America/Los_Angeles")
start = timezone.localize(datetime(2018, 9, 5))
end = timezone.localize(datetime(2018, 9, 6))
period = 5  # minute
voltage = 220  # volts
default_battery_power = 32 * voltage / 1000  # kW
site = "caltech"
signals = {'tariff': TimeOfUseTariff('sce_tou_ev_4_march_2019')}

# -- Network -----------------------------------------------------------------------------------------------------------
cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

# -- Events ------------------------------------------------------------------------------------------------------------
API_KEY = "DEMO_TOKEN"

# returns eventsqueue
# TODO: check if the event has stored the necessary info to reference it in sim
# we can reference session/ev in simulation by session.id, so find it

# sklada sa z plugin eventov, ktore dedia triedy, ktore obsahuju tieto obmedzenia
events = acnsim.acndata_events.generate_events(
    API_KEY, site, start, end, period, voltage, default_battery_power
)

num_samples = min(len(events), 10)
sampled_plugin_events = random.sample(events._queue, num_samples)
a = 0
# for ev in events._queue:

# random_plugin_events = np.random.choice(events._queue, size=num_samples, replace=False)


# -- Scheduling Algorithm ----------------------------------------------------------------------------------------------
# sch = EarliestDeadlineFirstAlgo(increment=1)
sch = SmoothedLLF(increment=1)
sch2 = algorithms.SortedSchedulingAlgo(algorithms.earliest_deadline_first)
sch3 = algorithms.SortedSchedulingAlgo(algorithms.least_laxity_first)

# -- Simulator ---------------------------------------------------------------------------------------------------------
sim = acnsim.Simulator(
    deepcopy(cn), sch, deepcopy(events), start, period=period, verbose=True, signals=signals
)
sim.run()

# For comparison we will also run the builtin earliest deadline first algorithm
sim2 = acnsim.Simulator(deepcopy(cn), sch2, deepcopy(events), start, period=period, signals=signals)
sim2.run()



sim3 = acnsim.Simulator(deepcopy(cn), sch3, deepcopy(events), start, period=period, signals=signals)
sim3.run()


simulations = [sim, sim2, sim3]

# -- Analysis ----------------------------------------------------------------------------------------------------------
# We can now compare the two algorithms side by side by looking that the plots of aggregated current.
# We see from these plots that our implementation matches th included one quite well. If we look closely however, we
# might see a small difference. This is because the included algorithm uses a more efficient bisection based method
# instead of our simpler linear search to find a feasible rate.
total_energy_prop = acnsim.proportion_of_energy_delivered(sim)
print("Proportion of requested energy delivered (Our sLLF): {0}".format(total_energy_prop))
total_energy_cost = acnsim.energy_cost(sim) + acnsim.demand_charge(sim)
print(f'Total energy cost (Our sLLF): {acnsim.energy_cost(sim)} + {acnsim.demand_charge(sim)} = {total_energy_cost}')

total_energy_prop = acnsim.proportion_of_energy_delivered(sim2)
total_energy_cost = acnsim.energy_cost(sim2) + acnsim.demand_charge(sim2)
print("Proportion of requested energy delivered (Their EADF): {0}".format( total_energy_prop))
print(f'Total energy cost (Their EADF): {acnsim.energy_cost(sim2)} + {acnsim.demand_charge(sim2)} = {total_energy_cost}')


total_energy_prop = acnsim.proportion_of_energy_delivered(sim3)
total_energy_cost = acnsim.energy_cost(sim3) + acnsim.demand_charge(sim3)
print("Proportion of requested energy delivered (Their LLF): {0}".format(total_energy_prop))
print(f'Total energy cost (Their LLF): {acnsim.energy_cost(sim3)} + {acnsim.demand_charge(sim3)} = {total_energy_cost}')


evs = []
for se in sampled_plugin_events:

    evs.append(sim.ev_history[se[1].session_id])










# Get list of datetimes over which the simulations were run.

# TODO: do it in for cycle
sim_dates = mdates.date2num(acnsim.datetimes_array(sim))
sim2_dates = mdates.date2num(acnsim.datetimes_array(sim2))
sim3_dates = mdates.date2num(acnsim.datetimes_array(sim3))


# Set locator and formatter for datetimes on x-axis.
locator = mdates.AutoDateLocator(maxticks=6)
formatter = mdates.ConciseDateFormatter(locator)



# TODO change ncol to num of simulations
fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
# axs[0].plot(sim_dates, acnsim.aggregate_current(sim), label="Our sLLF")
# axs[1].plot(sim2_dates, acnsim.aggregate_current(sim2), label="Their EDF")
# axs[2].plot(sim3_dates, acnsim.aggregate_current(sim2), label="Their LLF")

# TODO: check in what units maximum charging rate of ev is
# change it for another experiment
# axs[0].plot(sim_dates, acnsim.aggregate_power(sim), label="Our sLLF")
# axs[1].plot(sim2_dates, acnsim.aggregate_power(sim2), label="Their EDF")
# axs[2].plot(sim3_dates, acnsim.aggregate_power(sim2), label="Their LLF")

axs[0].plot(sim_dates, acnsim.aggregate_energy(sim), label="Our sLLF")
axs[1].plot(sim2_dates, acnsim.aggregate_energy(sim2), label="Their EDF")
axs[2].plot(sim3_dates, acnsim.aggregate_energy(sim2), label="Their LLF")


axs[0].set_title("Our sLLF")
axs[1].set_title("Their EDF")
axs[2].set_title("Their LLF")

# unit of energy is kWh
for ax in axs:
    # ax.set_ylabel("Current (A)")
    # ax.set_ylabel("Power  (kW)")
    ax.set_ylabel("Energy (kWh)")
    for label in ax.get_xticklabels():
        label.set_rotation(40)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

plt.show()


# TODO: add test for
# 2 cars arriving and departing at same time with same maximal charging rate

plot_charging_profiles(simulations=simulations, evs=evs)



timezone = pytz.timezone("America/Los_Angeles")
start = timezone.localize(datetime(2018, 9, 5))
end = timezone.localize(datetime(2018, 9, 6))
period = 5  # minute
voltage = 1e-10  # volts
default_battery_power =  32*voltage / 1000  # kW
site = "caltech"

# -- Network -----------------------------------------------------------------------------------------------------------
# cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)
evse_ids = ["CA-493","CA-496"]
cn = acnsim.sites.simple_acn(evse_ids,
                             evse_type="BASIC",
                             voltage=208,
                             # voltage=1,
                             # in kW
                             aggregate_cap=10)

# cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

common_arrival_time = 10
common_departure_time = 60

# arrival,
#         departure,
#         requested_energy,
#         station_id,
#         session_id,
#         battery,
#         estimated_departure=None,

ev1 = acnsim.EV(common_arrival_time, common_departure_time, 360, evse_ids[0], "EV-001", acnsim.Battery(100, 50, 5))
ev2 = acnsim.EV(common_arrival_time, common_departure_time, 360, evse_ids[1], "EV-002", acnsim.Battery(100, 50, 4))
evs = [ev1,
       ev2]

# charging rate will be slower than maximal charging rate, because evse max charging rate is smaller

plugin_event1 = PluginEvent(
common_arrival_time, ev1
)
plugin_event2 = PluginEvent(common_arrival_time, ev2)
# events = [PluginEvent(sess.arrival, sess) for sess in evs]
events = acnsim.EventQueue()
events.add_events([plugin_event1, plugin_event2])

sim4 = acnsim.Simulator(
    deepcopy(cn), sch, deepcopy(events), start, period=period, verbose=True
)
sim4.run()
# cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)
sim5 = acnsim.Simulator(
    deepcopy(cn), sch3, deepcopy(events), start, period=period, verbose=True
)
sim5.run()

num_samples = min(len(events), 10)
# sampled_plugin_events = random.sample(events._queue, num_samples)

simulations = [sim4, sim5]

# sampled_plugin_events = random.sample(events._queue, num_samples)
# evs = []
# for se in sampled_plugin_events:
#     evs.append(sim5.ev_history[se[1].session_id])




plot_charging_profiles(simulations=simulations, evs=evs)


