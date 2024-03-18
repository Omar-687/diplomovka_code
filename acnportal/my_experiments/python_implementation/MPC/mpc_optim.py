from typing import List, Union, Optional

import adacharge
from adacharge import AdaptiveChargingOptimization
import numpy as np
from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
import cvxpy as cp
from traitlets import Bool

class MPCOptimizer:
    def __init__(self):
        pass
    def solve(self,
              rates: cp.Variable,
              active_evs: List["SessionInfo"],
              objective,
              constraints,
              infrastructure,
              interface,
              optimisation_horizon,
              solver,
              verbose):
        # change rates to pilot_rates if they are really pilot rates
        scheduling_problem = cp.Problem(objective=cp.Maximize(objective),
                                        constraints=constraints)

        # TODO: adjust solver settings for better performance
        scheduling_problem.solve(solver=solver,
                                 feastol = 1e-2, reltol_inacc = 1e-5,
                                 verbose=verbose)

        if scheduling_problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return ValueError("Solver failed to solve the problem!")

        return rates.value


