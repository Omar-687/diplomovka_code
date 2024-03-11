from mpc_optim import *
import adacharge
import numpy as np
from mpc_sched import MyMpcAlgorithm
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy
from acnportal.signals.tariffs.tou_tariff import TimeOfUseTariff
from objective_enum import ObjectiveEnum
from acnportal import algorithms
from acnportal import acnsim
from mpc_optim import MPCOptimizer
from mpc_objective import  Objective
# test the code in python not jupyter, because of debugging

def test_functions():
    new_rates = np.zeros((3,3))
    # rates = [20, 30]
    MPCOptimizer.charging_rate_bounds(rates=new_rates,active_sessions=[],evse_indices=[])

def test1():
    # -- Experiment Parameters ---------------------------------------------------------------------------------------------
    timezone = pytz.timezone('America/Los_Angeles')
    start = timezone.localize(datetime(2018, 9, 5))
    end = timezone.localize(datetime(2018, 9, 6))
    period = 5  # minute
    voltage = 220  # volts
    default_battery_power = 32 * voltage / 1000  # kW
    site = 'caltech'
    signals = {'tariff': TimeOfUseTariff('sce_tou_ev_4_march_2019')}

    # -- Network -----------------------------------------------------------------------------------------------------------
    cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)



    # -- Events ------------------------------------------------------------------------------------------------------------
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)

    # my code
    obj = Objective()
    objective = [
        obj.choose_energy_maximisation_objective(),
        # obj.choose_consumer_cost_minimisation_objective()
        # obj.minimize_customers_chaging_costs(obj),
        # obj.charge_as_quickly_as_possible(obj),
        # smoothen_charging_rates
    ]
    sch_mpc = MyMpcAlgorithm(objective=objective,
                             verbose=False,
                             constraint_type='SOC',
                             solver=cp.ECOS)

    sim_mpc = acnsim.Simulator(deepcopy(cn),
                               sch_mpc,
                               deepcopy(events),
                               start,
                               period=period,
                               verbose=False,
                               signals=signals)
    sim_mpc.run()

    # testing of their code


    quick_charge_obj = [
        adacharge.ObjectiveComponent(adacharge.total_energy)
        # adacharge.ObjectiveComponent(adacharge.quick_charge),
        #                 adacharge.ObjectiveComponent(adacharge.equal_share, 1e-12)
    ]
    mpc = adacharge.AdaptiveSchedulingAlgorithm(quick_charge_obj, solver="ECOS", verbose=False)

    sim_mpc2 = acnsim.Simulator(deepcopy(cn),
                                mpc,
                                deepcopy(events),
                                start,
                                period=period,
                                verbose=False,
                                signals=signals)
    # sim_mpc2.run()

    simulations = [sim_mpc]
    print_results(network=cn,
                  simulations=simulations)

#     TODO: to find errors use his function build problem and substitute constraints or objective into solver
#    TODO: add adacharge here



def print_results(network ,simulations):
    cn = deepcopy(network)
    for sim_num, simulation in enumerate(simulations, start=1):
        sim = deepcopy(simulation)
        r = {'proportion_of_energy_delivered': acnsim.proportion_of_energy_delivered(sim),
             'energy_delivered': sum(ev.energy_delivered for ev in sim.ev_history.values()),
             # 'num_swaps': cn.swaps,
             # 'num_never_charged': cn.never_charged,
             'energy_cost': acnsim.energy_cost(sim),
             'demand_charge': acnsim.demand_charge(sim)
             }
        r['total_cost'] = r['energy_cost'] + r['demand_charge']
        r['$/kWh'] = r['total_cost'] / r['energy_delivered']
        print(f'Simulation number: {sim_num}')
        print(r)
        print()
        print()
def run():
    test1()
    # test_functions()

run()