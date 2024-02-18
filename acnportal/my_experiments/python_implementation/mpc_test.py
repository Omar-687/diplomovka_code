from mpc_optim import *
import adacharge
import numpy as np

def test_functions():
    new_rates = np.zeros((3,3))
    # rates = [20, 30]
    MPCOptimizer.charging_rate_bounds(rates=new_rates,active_sessions=[],evse_indices=[])

def test1():
    objective = {
        adacharge.total_energy: 1000,
        adacharge.tou_energy_cost:None,
        adacharge.quick_charge: 1e-6,


    }

    MPCOptimizer(objective=objective, constraint_type="SOC", interface=None)

def run():
    test1()
    test_functions()

run()