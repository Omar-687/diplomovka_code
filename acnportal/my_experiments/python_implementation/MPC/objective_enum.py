from enum import Enum


class ObjectiveEnum(Enum):
    QUICK_CHARGING = 'quick_charging'
    CUSTOMER_PROFIT = 'customer_cost_minimisation'
    ENERGY_MAXIMISATION = 'energy_maximisation'

    # TODO: add more objectives