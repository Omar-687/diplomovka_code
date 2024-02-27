from acnportal.algorithms import BaseAlgorithm
import warnings

class MyMpcAlgorithm(BaseAlgorithm):
    def __init__(self,
                 objective,
                 constraint_type="SOC",
                 # enforce_energy_equality=False,
                 # peak_limit=None,
                 # estimate_max_rate=False,
                 max_rate_estimator=None,
                 # uninterrupted_charging=False,
                 # quantize=False,
                 # reallocate=False,
                 max_recompute=None,
                 allow_overcharging=False,
                 # verbose=False
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
        # self.enforce_energy_equality = enforce_energy_equality
        # # self.solver = solver
        # self.peak_limit = peak_limit
        # self.estimate_max_rate = estimate_max_rate
        self.max_rate_estimator = max_rate_estimator
        # self.uninterrupted_charging = uninterrupted_charging
        # self.quantize = quantize
        # self.reallocate = reallocate
        # self.verbose = verbose
        # if not self.quantize and self.reallocate:
        #     raise ValueError(
        #         "reallocate cannot be true without quantize. "
        #         "Otherwise there is nothing to reallocate :)."
        #     )
        # if self.quantize:
        #     if self.max_recompute is not None:
        #         warnings.warn(
        #             "Overriding max_recompute to 1 " "since quantization is on."
        #         )
        #     self.max_recompute = 1
        # else:
        #     self.max_recompute = max_recompute
        # self.allow_overcharging = allow_overcharging

    def register_interface(self, interface):
        """Register interface to the _simulator/physical system.

        This interface is the only connection between the algorithm and what it
            is controlling. Its purpose is to abstract the underlying
            network so that the same algorithms can run on a simulated
            environment or a physical one.

        Args:
            interface (Interface): An interface to the underlying network
                whether simulated or real.

        Returns:
            None
        """
        self._interface = interface
        if self.max_rate_estimator is not None:
            self.max_rate_estimator.register_interface(interface)

    def schedule(self, active_sessions):
        if len(active_sessions) == 0:
            return {}
        infrastructure = self.interface.infrastructure_info()

        active_sessions = enforce_pilot_limit(active_sessions, infrastructure)
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