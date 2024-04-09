import subprocess
import sys
import pytz
from datetime import datetime
import numpy as np
# import cvxpy as cp
from copy import deepcopy
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from acnportal.signals.tariffs.tou_tariff import TimeOfUseTariff
from acnportal import acnsim
from acnportal.acnsim import analysis
from acnportal import algorithms
import adacharge


def single_phase_caltech_acn(basic_evse=False, voltage=208, transformer_cap=150, network_type=acnsim.ChargingNetwork):
    """ Predefined single phase ChargingNetwork for the Caltech ACN.

    Args:
        basic_evse (bool): If True use BASIC EVSE type instead of actual AeroViroment and ClipperCreek types.
        voltage (float): Default voltage at the EVSEs. Does not affect the current rating of the transformer which is
            based on nominal voltages in the network. 277V LL for delta primary and 120V LN for wye secondary. [V]
        transformer_cap (float): Capacity of the transformer in the CaltechACN. Default: 150. [kW]

    Attributes:
        See ChargingNetwork for Attributes.
    """
    network = network_type()

    if basic_evse:
        evse_type = {'AV': 'BASIC', 'CC': 'BASIC'}
    else:
        evse_type = {'AV': 'AeroVironment', 'CC': 'ClipperCreek'}

    # Define the sets of EVSEs in the Caltech ACN.
    CC_pod_ids = ["CA-322", "CA-493", "CA-496", "CA-320", "CA-495", "CA-321", "CA-323", "CA-494"]
    AV_pod_ids = ["CA-324", "CA-325", "CA-326", "CA-327", "CA-489", "CA-490", "CA-491", "CA-492"]
    other_ids = [f"CA-{id_num}" for id_num in [148, 149, 212, 213, 303, 304, 305, 306, 307, 308,
                                               309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
                                               319, 497, 498, 499, 500, 501, 502, 503, 504, 505,
                                               506, 507, 508, 509, 510, 511, 512, 513]]
    all_ids = CC_pod_ids + AV_pod_ids + other_ids

    # Add Caltech EVSEs
    for evse_id in all_ids:
        if evse_id not in CC_pod_ids:
            network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type['AV']), voltage, 0)
        else:
            network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type['CC']), voltage, 0)

    # Add Caltech Constraint Set
    CC_pod = acnsim.Current(CC_pod_ids)
    AV_pod = acnsim.Current(AV_pod_ids)
    all_current = acnsim.Current(all_ids)

    # Build constraint set
    network.add_constraint(CC_pod, 80, name='CC Pod')
    network.add_constraint(AV_pod, 80, name='AV Pod')
    network.add_constraint(all_current, transformer_cap * 1000 / voltage, name='Transformer Cap')
    return network


def experiment(algorithm):
    """ Run single phase vs. three phase experiment for a particular algorithm. """
    # -- Experiment Parameters ---------------------------------------------------
    timezone = pytz.timezone('America/Los_Angeles')
    start = timezone.localize(datetime(2018, 9, 5))
    end = timezone.localize(datetime(2018, 9, 6))
    period = 5  # minute
    voltage = 208  # volts
    default_battery_power = 32 * voltage / 1000  # kW
    site = 'caltech'
    signals = {'tariff': TimeOfUseTariff('sce_tou_ev_4_march_2019')}
    # -- Network -------------------------------------------------------------------
    single_phase_cn = single_phase_caltech_acn(basic_evse=True, transformer_cap=70)
    real_cn = acnsim.sites.caltech_acn(basic_evse=True, transformer_cap=70)

    # -- Events ---------------------------------------------------------------------
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period,
                                                   voltage, default_battery_power)

    # -- Single Phase ----------------------------------------------------------------
    single_phase_sim = acnsim.Simulator(deepcopy(single_phase_cn), algorithm,
                                        deepcopy(events), start, period=period,
                                        verbose=False,signals=signals)
    single_phase_sim.run()

    # Since we are interested in how the single-phase LLF algorithm would have performed
    # in the real CaltechACN, we replace the network model with the real network model
    # for analysis.
    single_phase_sim.network = real_cn

    # -- Three Phase -----------------------------------------------------------------
    three_phase_sim = acnsim.Simulator(deepcopy(real_cn), algorithm,
                                       deepcopy(events), start, period=period,
                                       verbose=False, signals=signals)
    three_phase_sim.run()

    return single_phase_sim, three_phase_sim


# llf = algorithms.SortedSchedulingAlgo(algorithms.least_laxity_first)
# llf_sp_sim, llf_tp_sim = experiment(llf)

quick_charge_obj = [
        adacharge.ObjectiveComponent(adacharge.tou_energy_cost)
    # adacharge.ObjectiveComponent(adacharge.quick_charge),
                    # adacharge.ObjectiveComponent(adacharge.equal_share, 1e-12),
                    # adacharge.ObjectiveComponent(adacharge.non_completion_penalty, 1e-12),
                    ]
mpc = adacharge.AdaptiveSchedulingAlgorithm(quick_charge_obj, solver="ECOS")
mpc_sp_sim, mpc_tp_sim = experiment(mpc)



total_energy_prop = acnsim.proportion_of_energy_delivered(mpc_tp_sim)
print('Proportion of requested energy delivered: {0}'.format(total_energy_prop))

print('Peak aggregate current: {0} A'.format(mpc_tp_sim.peak))

# Plotting aggregate current
agg_current = acnsim.aggregate_current(mpc_tp_sim)
plt.plot(agg_current)
plt.xlabel('Time (periods)')
plt.ylabel('Current (A)')
plt.title('Total Aggregate Current')
plt.show()


