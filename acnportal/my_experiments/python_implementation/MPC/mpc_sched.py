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
                 # enforce_energy_equality=False,
                 # peak_limit=None,
                 # estimate_max_rate=False,
                 # max_rate_estimator=None,
                 solver=cp.ECOS,
                 # uninterrupted_charging=False,
                 # quantize=False,
                 # reallocate=False,
                 # max_recompute=None,
                 # allow_overcharging=False,
                 verbose=False
                 ):
        """Model Predictive Control based Adaptive Schedule Algorithm compatible with BaseAlgorithm.

                Args:
                    objective (List[ObjectiveComponent]): List of ObjectiveComponents
                        for the optimization.
                    constraint_type (str): String representing which constraint type
                        to use. Options are 'SOC' for Second Order Cone or 'LINEAR'
                        for linearized constraints.
                    enforce_energy_equality (bool): If True, energy delivered must
                        be  equal to energy requested for each EV. If False, energy
                        delivered must be less than or equal to request.
                    solver (str): Backend solver to use. See CVXPY for available solvers.
                    peak_limit (Union[float, List[float], np.ndarray]): Limit on
                    aggregate peak current. If None, no limit is enforced.
                    rampdown (Rampdown): Rampdown object used to predict the maximum
                        charging rate of the EV's battery. If None, no ramp down is
                        applied.
                    minimum_charge (bool): If true EV should charge at least at the
                        minimum non-zero charging rate of the EVSE it is connected
                        to for the first control period.
                    quantize (bool): If true, apply project_into_discrete_feasible_pilots post-processing step.
                    reallocate (bool): If true, apply index_based_reallocation
                        post-processing step.
                    max_recompute (int): Maximum number of control periods between
                        optimization solves.
                    allow_overcharging (bool): Allow the algorithm to exceed the energy
                        request of the session by at most the energy delivered at the
                        minimum allowable rate for one period.
                    verbose (bool): Solve with verbose logging. Helpful for debugging.
                """
        super().__init__()
        self.objective = objective
        self.constraint_type = constraint_type
        # self.max_rate_estimator = max_rate_estimator
        self.objective_list = objective
        if solver not in cp.installed_solvers():
            raise ValueError('Chosen solver is not cvxpy solver!')

        self.solver = solver
        self.verbose = verbose

    def schedule(self, active_sessions:List['SessionInfo']):
        """ See BaseAlgorithm """
        if len(active_sessions) == 0:
            schedule = {}
            return schedule
        infrastructure:InfrastructureInfo = self.interface.infrastructure_info()
        active_sessions = enforce_pilot_limit(active_sessions, infrastructure)
        self.peak_limit = None
        self.objective_list =  [
            adacharge.ObjectiveComponent(adacharge.total_energy)
            # adacharge.ObjectiveComponent(adacharge.quick_charge),
            #                 adacharge.ObjectiveComponent(adacharge.equal_share, 1e-12)
        ]
        optimizer = AdaptiveChargingOptimization(
            self.objective_list,
            self.interface,
            self.constraint_type,
            # self.enforce_energy_equality,
            solver=self.solver,
        )
        rates_matrix = optimizer.solve(
            active_sessions,
            infrastructure,
            peak_limit=None,
            prev_peak=self.interface.get_prev_peak(),
            verbose=self.verbose
        )


        # structuring output
        schedule = {}
        for station_id in infrastructure.station_ids:
            # rates is shape (N,T), N= number of stations,T optimisation horizon
            # necessary to find a assigned row for EVSE
            i = infrastructure.get_station_index(station_id=station_id)
            schedule[station_id] = list(rates_matrix[i])
        # assuming the order of cars with their charging rates is same as in active_evs
        # charging_rates_for_prediction_horizon = mpc_optimizer.solve()
        return schedule


    def schedule2(self, active_evs: List["SessionInfo"]):
        # add typing so i can access functions from acnportal
        schedule = {}
        if len(active_evs) == 0:
            return schedule
        infrastructure:InfrastructureInfo = self.interface.infrastructure_info()
        A = 0

        active_evs = enforce_pilot_limit(active_sessions=active_evs, infrastructure=infrastructure)
        # optimisation_horizon_T_threshold = max([active_ev.remaining_time for active_ev in active_evs])
        # optimisation_horizon_T = [i for i in range(1, optimisation_horizon_T_threshold + 1)]

        optimisation_horizon_T = max(
            s.arrival_offset + s.remaining_time for s in active_evs)


        # (54,T)
        rates = cp.Variable(shape=(infrastructure.num_stations, optimisation_horizon_T))
        # generated_objective = Objective().generate_objective(rates=rates,
        #                                                      objective_list=self.objective_list,
        #                                                      infrastructure=infrastructure,
        #                                                      interface=self.interface,
        #                                                      optimisation_horizon_T=optimisation_horizon_T
        #                                                      )
        # generate_constraints = MPCConstraintsGenerator().get_all_constraints(rates=rates,
        #                                                                      active_evs=active_evs,
        #                                                                      infrastructure=infrastructure,
        #                                                                      optimisation_horizon_T=optimisation_horizon_T,
        #                                                                      constraint_type=self.constraint_type,
        #                                                                      delta_period=5,
        #                                                                      minimal_value_charging=False
        #
        #                                                                      )
        generate_constraints = []
        generated_objective = []
        rates = MPCOptimizer().solve(rates=rates,
                                     active_evs=active_evs,
                                     objective=generated_objective,
                                     constraints=generate_constraints,
                                     optimisation_horizon_T=optimisation_horizon_T,
                                     infrastructure=infrastructure,
                                     interface=self.interface,
                                     solver=self.solver,
                                     verbose=self.verbose)

        # rates = project_into_continuous_feasible_pilots(
        #      rates, infrastructure
        # )
        #
        # # converts charging rates to zeros
        # rates = np.maximum(rates, 0)
        # .solve(active_evs=active_evs,
        #                                      objective=
        #                                      optimisation_horizon_T=optimisation_horizon_T,
        #                                      )
        for active_ev in active_evs:
            # rates is shape (N,T), N= number of cars,T optimisation horizon
            ...
            # schedule[active_ev.station_id] = [rates[i,0]]
        # assuming the order of cars with their charging rates is same as in active_evs
        # charging_rates_for_prediction_horizon = mpc_optimizer.solve()

        return schedule


        # for i, active_session in enumerate(active_sessions):
        #     ...

        # active_sessions = enforce_pilot_limit(active_sessions, infrastructure)
        # MPC


# For example, the problem of maximizing a concave function
# �
# {\displaystyle f} can be re-formulated equivalently as the problem of minimizing the convex function
# −
# �

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

