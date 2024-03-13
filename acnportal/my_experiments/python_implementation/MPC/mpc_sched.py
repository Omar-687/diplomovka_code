import numpy as np
from acnportal.algorithms import BaseAlgorithm
from mpc_optim import  *
import cvxpy as cp
import numpy
from objective_enum import ObjectiveEnum
from mpc_objective import Objective
from mpc_constraints_generator import MPCConstraintsGenerator
from acnportal.algorithms.preprocessing import *
import warnings
from postprocessing import (
    project_into_continuous_feasible_pilots,
    project_into_discrete_feasible_pilots,
)
from postprocessing import index_based_reallocation, diff_based_reallocation
class MyMpcAlgorithm(BaseAlgorithm):
    def __init__(self,
                 objective,
                 constraint_type="SOC",
                 solver=cp.ECOS,
                 verbose=False
                 ):

        super().__init__()
        self.objective = objective

        # TODO: checking if all objectives in array are turned off
        # for obj in self.objective:
        #
        self.constraint_type = constraint_type
        # self.max_rate_estimator = max_rate_estimator
        # self.objective_list = objective
        if solver not in cp.installed_solvers():
            raise ValueError('Chosen solver is not cvxpy solver!')

        self.solver = solver
        self.verbose = verbose

    def schedule(self, active_sessions:List['SessionInfo']):
        """ See BaseAlgorithm """
        if len(active_sessions) == 0:
            schedule = {}
            return schedule

        # pre-processing
        infrastructure:InfrastructureInfo = self.interface.infrastructure_info()
        active_sessions = enforce_pilot_limit(active_sessions, infrastructure)


        # algorithm
        # define optimisation horizon and optimisation variable for utility function U_k
        max_remaining_time = max([session.remaining_time for session in active_sessions])
        optimisation_horizon = [i for i in range(1, max_remaining_time)]
        # pilot rates are in Amp, to get charging rates we need
        # to calculate pilot_rates * voltage of network
        pilot_rates = cp.Variable(shape=(infrastructure.num_stations,
                                   len(optimisation_horizon) - 1))

        # define constraints in cvxpy for feasible set R_k
        constraint_generator = MPCConstraintsGenerator()
        all_constraints = constraint_generator.get_all_constraints(rates=pilot_rates,
                                                 active_evs=active_sessions,
                                                 infrastructure=infrastructure,
                                                 constraint_type=self.constraint_type,
                                                 optimisation_horizon=optimisation_horizon,
                                                 period_between_timesteps=self.interface.period)

        # define utility function U_k from objective in cvxpy

        objective_generator = Objective()
        objective = objective_generator.generate_objective(rates=pilot_rates,
                                                           objective_list=self.objective,
                                                           infrastructure=infrastructure,
                                                           interface=self.interface,
                                                           optimisation_horizon=optimisation_horizon
                                                           )

        # use optimizer to solve the problem
        optimizer = MPCOptimizer()


        optimised_pilot_rates = optimizer.solve(rates=pilot_rates,
                        active_evs=active_sessions,
                        objective=objective,
                        constraints=all_constraints,
                        infrastructure=infrastructure,
                        interface=self.interface,
                        optimisation_horizon=optimisation_horizon,
                        solver=self.solver,
                        verbose=self.verbose)

        # structuring output
        schedule = {}
        for station_id in infrastructure.station_ids:
            i = infrastructure.get_station_index(station_id=station_id)
            schedule[station_id] = list(optimised_pilot_rates[i])

        return schedule



#  second order cone
# 1. LP (linear programming), 2. Quadratic programming (QP), 3. Second order cone (SOCP)
# SOCPs can be solved by interior point methods
# The "second-order cone" in SOCP arises from the constraints,

# A convex optimization problem is a problem where all of the constraints are convex functions, and the objective is a convex function if minimizing, or a concave function if maximizing.
# convex problems can be linear or nonlinear, since linear programming is smallest subset of convex problems


# they used MOSEK and ECOS solver, it says that they are used for convex optim second order problems

# Linear programming problems are the simplest convex programs. In LP, the objective and constraint functions are all linear.

# {\displaystyle -f}. The problem of maximizing a concave function over a convex set is commonly called a convex optimization problem
# Convex optimization is a subfield of mathematical optimization that studies the problem of minimizing convex functions over convex sets (or, equivalently, maximizing concave functions over convex sets)

