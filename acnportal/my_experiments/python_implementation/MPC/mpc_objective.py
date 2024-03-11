from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
from objective_enum import ObjectiveEnum
import cvxpy as cp
from typing import List, Union, Optional, Any
import numpy as np
class Objective:
    def __init__(self,
                 # infrastructure:InfrastructureInfo,
                 # interface:Interface

                 ):
        # self.infrastructure:InfrastructureInfo = {}
        # self.all_chosen_objectives = {}
        # self.infrastructure = infrastructure
        # self.objectives_arr = []
        pass



    # generate objective

    # TODO: is it written correctly? - objective_list:List[ObjectiveEnum, Any]
    # TODO: all functions here must share input arguments so we can ease the for cycle
    def generate_objective(self,
                           rates: cp.Variable,
                           objective_list,
                           infrastructure: InfrastructureInfo,
                           interface: Interface,
                           optimisation_horizon_T):
        res = 0
        for objection in objective_list:
            objection_name, objection_relative_weight, objection_function = objection
            res += objection_function(rates=rates,
                                      relative_weight=objection_relative_weight,
                                      infrastructure=infrastructure,
                                      interface=interface,
                                      optimisation_horizon_T=optimisation_horizon_T)
        return res
    #     objective functions
    def choose_energy_maximisation_objective(self, relative_weight=1):
        # relative weight must be above 0, a > 0, where weight is a
        if relative_weight <= 0:
            raise ValueError("Relative weight of objective must be positive!")
        return ObjectiveEnum.ENERGY_MAXIMISATION.value, relative_weight, self.maximize_charging_energy

    def charging_power(self,rates, infrastructure, **kwargs):
        """ Returns a matrix with the same shape as rates but with units kW instead of A. """
        voltage_matrix = np.tile(infrastructure.voltages, (rates.shape[1], 1)).T
        return cp.multiply(rates, voltage_matrix) / 1e3
    def get_period_energy(self,rates, infrastructure, period, **kwargs):
        """ Return energy delivered in kWh during each time period and each session. """
        power = self.charging_power(rates, infrastructure=infrastructure)
        period_in_hours = period / 60
        return power * period_in_hours

    def total_energy(self,rates, infrastructure, interface, **kwargs):
        return cp.sum(self.get_period_energy(rates, infrastructure, interface.period))

    def maximize_charging_energy(self,
                                 rates: cp.Variable,
                                 relative_weight,
                                 infrastructure: InfrastructureInfo,
                                 interface: Interface,
                                 optimisation_horizon_T):
        # obj = None
        # self.objectives_arr.append(obj)
        return self.total_energy(rates,infrastructure,interface)
        # return relative_weight * cp.sum(rates)

    #
    def choose_consumer_cost_minimisation_objective(self,relative_weight=1):
        # relative weight must be above 0, a > 0, where weight is a
        if relative_weight <= 0:
            raise ValueError("Relative weight of objective must be positive!")
        # cost minimization and profit maximisation are equivalent problems
        return ObjectiveEnum.CUSTOMER_PROFIT.value, relative_weight, self.minimize_customers_chaging_costs
    def minimize_customers_chaging_costs(self,
                                         rates: cp.Variable,
                                         relative_weight,
                                         infrastructure: InfrastructureInfo,
                                         interface: Interface,
                                         optimisation_horizon_T
                                         ):
        current_time = interface.current_time
        energy_costs = interface.get_prices(length=rates.shape[0],
                                            start=current_time)

        # TODO:dont use * when multiplying matrices
        return relative_weight*cp.sum(energy_costs * rates)



    # TODO implement also demand charge costs later on
    # but better is to test basic functionality and later finish it


    # Regularizers
    def choose_quick_charging_objective(self, relative_weight=1):
        if relative_weight <= 0:
            raise ValueError("Relative weight of objective must be positive!")
        return ObjectiveEnum.QUICK_CHARGING.value, relative_weight, self.charge_as_quickly_as_possible
    def charge_as_quickly_as_possible(self,
                                      rates: cp.Variable,
                                      relative_weight,
                                      infrastructure: InfrastructureInfo,
                                      interface: Interface,
                                      optimisation_horizon_T

                                      ):

        # here we assume optim horizon is array of ints
        # where to get current time??? from interface
        util_value = 0
        for t in optimisation_horizon_T:
            util_value += ((optimisation_horizon_T[-1] - t + 1)
            / optimisation_horizon_T[-1]) * cp.sum(rates[:, t])
        return relative_weight*util_value

    # what it exactly is?
    def equal_share(self,
                    rates: cp.Variable,
                    relative_weight,
                    infrastructure: InfrastructureInfo,
                    interface: Interface,
                    optimisation_horizon_T
                    ):

        # from https://www.ncl.ac.uk/media/wwwnclacuk/cesi/files/20190814_Adaptive%20Charging%20Network_webinar.pdf presentation

        # TODO: is cp.square same as np.square?
        return -cp.sum(cp.square(rates))




