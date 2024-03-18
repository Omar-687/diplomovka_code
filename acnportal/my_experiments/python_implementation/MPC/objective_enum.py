from enum import Enum

# TODO: try his code for OSPQ for the test
# TODO: try to fix load flattening problem too
# TODO: add smoothing
# adacharge/adacharge/tests
# /test_adaptive_charging_optimization.py

class ObjectiveEnum(Enum):
    QUICK_CHARGING = 'quick_charging'
    CUSTOMER_PROFIT = 'customer_cost_minimisation'
    ENERGY_MAXIMISATION = 'energy_maximisation'
    EQUAL_SHARE = 'equal_share'
    SMOOTHING = 'smoothing'

    # TODO: add more objectives