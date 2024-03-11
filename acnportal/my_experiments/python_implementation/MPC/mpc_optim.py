from typing import List, Union, Optional

import adacharge
from adacharge import AdaptiveChargingOptimization
import numpy as np
from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
import cvxpy as cp
from traitlets import Bool

class MPCOptimizer:
    def __init__(self,
                 # active_evs,
                 # feasible_set,
                 # objective_set,
                 # optimisation_horizon_T,
                 # infrastructure,
                 # constraint_type,
                 # solver
                 ):
        pass
        # self.solver = solver
        # self.objective = objective_set
        # self.optimisation_horizon_T = optimisation_horizon_T
        # self.constraint_type = constraint_type
        # self.infrastructure = infrastructure
        # super().__init__(interface, objective)


    def solve(self,
              rates,
              active_evs: List["SessionInfo"],
              objective,
              constraints,
              infrastructure,
              interface,
              optimisation_horizon_T,
              solver,
              verbose):
        quick_charge_obj = [
            adacharge.ObjectiveComponent(adacharge.total_energy)
            # adacharge.ObjectiveComponent(adacharge.quick_charge),
            #                 adacharge.ObjectiveComponent(adacharge.equal_share, 1e-12)
        ]
        abc = AdaptiveChargingOptimization(objective=quick_charge_obj,interface=interface)
        # built_prob = abc.build_problem(active_sessions=active_evs,infrastructure=infrastructure)
        #
        # problem = cp.Problem(
        #     built_prob["objective"], list(built_prob["constraints"].values())
        # )
        # problem.solve(solver=solver, verbose=verbose)
        #
        # if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        #     return ValueError("Solver failed to solve the problem!")

        # some predictions were below 0
        # charging_rates = rates.value
        # charging_rates[charging_rates < 0] = 0
        return abc.solve(active_sessions=active_evs,infrastructure=infrastructure)
        # return built_prob["variables"]["rates"].value


