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
from adacharge.adaptive_charging_optimization import smoothing
import pandas as pd
def test_functions():
    new_rates = np.zeros((3,3))
    # rates = [20, 30]
    MPCOptimizer.charging_rate_bounds(rates=new_rates,active_sessions=[],evse_indices=[])


def setup(voltage=100, period=5):
    # -- Experiment Parameters ---------------------------------------------------------------------------------------------
    timezone = pytz.timezone('America/Los_Angeles')
    start = timezone.localize(datetime(2018, 9, 4))
    end = timezone.localize(datetime(2018, 9, 5))

    # period should be small number, otherwise we optimise only few timesteps
    period = period  # minute
    # lowering voltage will cause slower electric current flow
    # therefore less energy will be provided to ev
    voltage = voltage  # volts
    default_battery_power = 32 * voltage / 1000  # kW
    site = 'caltech'
    signals = {'tariff': TimeOfUseTariff('sce_tou_ev_4_march_2019')}

    # -- Network -----------------------------------------------------------------------------------------------------------

    # TODO: to make it work also for non basic evse, we need to add projection of rates into feasible pilot signals
    cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

    # -- Events ------------------------------------------------------------------------------------------------------------
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)
    return cn, start, period, signals, events

def test1():
    cn, start, period, signals, events = setup()
    print()
    print('Num of ev arrivals: ', len(events))
    # my code
    obj = Objective()
    objective = [
        # obj.choose_energy_maximisation_objective(turn_off_objective=True),
        obj.choose_energy_maximisation_objective(turn_off_objective=True),
        obj.choose_quick_charging_objective(turn_off_objective=True)
        # obj.choose_consumer_cost_minimisation_objective()
        # obj.minimize_customers_chaging_costs(obj),
        # obj.charge_as_quickly_as_possible(obj),
        # smoothen_charging_rates
    ]
    sch_mpc = MyMpcAlgorithm(objective=objective,
                             constraint_type='SOC',
                             solver=cp.ECOS,
                             verbose=False,)

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
        adacharge.ObjectiveComponent(adacharge.total_energy),
        adacharge.ObjectiveComponent(adacharge.quick_charge),
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


def test2():
    cn, start, period, signals, events = setup()
    print()
    print('Num of ev arrivals: ', len(events))
    # my code
    obj = Objective()
    objective = [
        # obj.choose_energy_maximisation_objective(turn_off_objective=True),
        obj.choose_energy_maximisation_objective(turn_off_objective=True),
        obj.choose_quick_charging_objective(turn_off_objective=True),
        # obj.choose_smoothing_objective(turn_off_objective=True)
        # equal share seems to cause warnings, check how to fix
        # obj.choose_equal_share_objective(turn_off_objective=True)
        # obj.choose_consumer_cost_minimisation_objective()
        # obj.minimize_customers_chaging_costs(obj),
        # obj.charge_as_quickly_as_possible(obj),
        # smoothen_charging_rates
    ]
    sch_mpc = MyMpcAlgorithm(objective=objective,
                             constraint_type='SOC',
                             solver=cp.ECOS,
                             verbose=False,)

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
        adacharge.ObjectiveComponent(adacharge.total_energy),
        adacharge.ObjectiveComponent(adacharge.quick_charge),
        # adacharge.ObjectiveComponent(adacharge.smoothing)

        # adacharge.ObjectiveComponent(adacharge.equal_share)
    ]
    mpc = adacharge.AdaptiveSchedulingAlgorithm(quick_charge_obj, solver="ECOS", verbose=False,constraint_type='LINEAR')

    sim_mpc2 = acnsim.Simulator(deepcopy(cn),
                                mpc,
                                deepcopy(events),
                                start,
                                period=period,
                                verbose=False,
                                signals=signals)
    # sim_mpc2.run()

    simulations = [sim_mpc]
    labels = [

        'Our MPC'
    ]
    print_table_results(network=cn,
                  all_simulations=simulations,labels=labels)



def test_solvers():
    cn, start, period, signals, events = setup()
    print()
    print('Num of ev arrivals: ', len(events))

    testing_solvers = [cp.ECOS, cp.CLARABEL]

    for i in range(len(testing_solvers)):
        test2()


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

def print_table_results(network, all_simulations, labels):


    # Data pro tabulku
    data = {
        'Simulation name': [],
        'Proportion of energy delivered': [],
        'Energy costs': [],
        'Energy delivered': [],
        'Demand Charge': [],
        'Total costs': []
        # 'Number of evs never charged': [],
        # 'Number of swaps': []
    }

    for i, simulation in enumerate(all_simulations):
        data['Simulation name'].append(labels[i])
        data['Proportion of energy delivered'].append(acnsim.proportion_of_energy_delivered(simulation))
        data['Energy delivered'].append(sum(ev.energy_delivered for ev in all_simulations[i].ev_history.values()))
        data['Energy costs'].append(acnsim.energy_cost(all_simulations[i]))
        data['Demand Charge'].append(acnsim.demand_charge(all_simulations[i]))
        data['Total costs'].append(data['Demand Charge'][-1] + data['Energy costs'][-1])
        # so far we test on ChargingNetwork
        # data['Number of evs never charged'].append(number_of_evs_never_charged_per_sim[i])
        # data['Number of swaps'].append(number_of_swaps_per_sim[i])

    # Vytvoření DataFrame
    df = pd.DataFrame(data)

    # Set pandas options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Vytisknutí tabulky
    print(df)


def run():
    # test1()
    test2()

    # test_functions()

run()