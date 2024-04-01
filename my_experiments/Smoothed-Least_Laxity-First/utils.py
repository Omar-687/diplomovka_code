import numpy as np
from acnportal.acnsim.interface import InfrastructureInfo
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from acnportal.acnsim.simulator import Simulator


def plot_charging_profiles(simulations:list[Simulator], evs, period=5):
    for simulation in simulations:
        fig1, axs1 = plt.subplots(len(evs), figsize=(10, len(evs) * 3), sharey=True)
        for plot_idx, ev in enumerate(evs, start=0):
            evse_index = simulation.network.station_ids.index(ev.station_id)
            session_len = ev.departure - ev.arrival
            x = [simulation.start + timedelta(minutes=period * ev.arrival) + timedelta(minutes=period * i) for i in range(session_len)]
            # charging_profile = simulation.charging_rates[evse_index][ev.arrival:ev.departure]
            # TODO: if there are more voltages simulation.network._voltages[evse_index], check if * is right operation

            charging_profile = (simulation.charging_rates[evse_index][ev.arrival:ev.departure] * simulation.network._voltages[evse_index] ) / (1000)
            charging_profile = charging_profile * (simulation.period/60)
            axs1[plot_idx].plot(x, charging_profile)
            plt.xlabel('Time [K]')
            # simulation.it
            # plt.yticks([0, 16, 32])
            plt.ylabel('Energy [kWh]')
        plt.show()
    # Some example data to display
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)
    # x1 = np.linspace(0, 6 * np.pi, 600)
    # y1 = np.sin(x1)
    # fig, axs = plt.subplots(3, sharey=True)
    # fig.suptitle('Sharing both axes')
    # axs[0].plot(x, y ** 2)
    # axs[1].plot(x1, 0.3 * y1, 'o')
    # axs[2].plot(x, y, '+')
    # plt.show()
# plot_charging_profiles([],[])


def infrastructure_constraints_feasible(rates, infrastructure: InfrastructureInfo):
    phase_in_rad = np.deg2rad(infrastructure.phases)
    for j, v in enumerate(infrastructure.constraint_matrix):
        a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
        line_currents = np.linalg.norm(a @ rates, axis=0)
        if not np.all(line_currents <= infrastructure.constraint_limits[j] + 1e-7):
            return False
    return True