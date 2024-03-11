import numpy as np
import cvxpy as cp
import numpy
from objective_enum import ObjectiveEnum
import warnings
from typing import List, Union, Optional
from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
from adacharge.adaptive_charging_optimization import AdaptiveChargingOptimization
from traitlets import Bool
class MPCConstraintsGenerator:
    def __init__(self):
        pass
    def get_all_constraints(self,
                            rates:cp.Variable,
                            active_evs: List["SessionInfo"],
                            infrastructure: InfrastructureInfo,
                            optimisation_horizon_T,
                            constraint_type,
                            delta_period:int,
                            minimal_value_charging=False):
        # is convex or is concave good functions to try out


        # variable constant or parameter can be at most 2D
        # similar semantics as numpy
        def charging_rate_interval_constraints():
            '''
            TODO: should i write description of method below or above params?
            # write it ABOVE, otherwise it will be in return section

            3a) constraint in the book Large-Scale Adaptive Electric Vehicle Charging.

            '''

            # TODO: lower bound can be 0 for each element i guess
            # TODO set up that values of lower bounds and upper bounds arrays are >= 0
            # rows, cols = infrastructure.num_stations, len(optimisation_horizon_T)
            # num_cars, optim_horizon = rates.shape[0], rates.shape[1]
            # lower_charging_bounds =  np.zeros((rows, cols))
            # upper_charging_bounds = np.zeros((rows, cols))
            lower_charging_bounds =  np.zeros(rates.shape)
            upper_charging_bounds = np.zeros(rates.shape)

            for active_ev in active_evs:
                # pozriet na priklady session and station id
                row_index:int = infrastructure.get_station_index(station_id=active_ev.station_id)


                # TODO: check if max and min rate should be taken from evse class or from ev class
                # TODO: use numpy instead of parameter cvxpy ?
                max_rates = active_ev.max_rates

                # if an ev leaves before ending of optimisation horizon


                upper_charging_bounds[row_index,
                active_ev.arrival_offset: active_ev.arrival_offset + active_ev.remaining_time ] = max_rates

                if minimal_value_charging:
                    lower_charging_bounds[row_index,
                    active_ev.arrival_offset: active_ev.arrival_offset + active_ev.remaining_time] = active_ev.min_rates
            return lower_charging_bounds <= rates, rates <= upper_charging_bounds





        def requested_energy_charged_constraints():
            '''
            3b) constraint in the publication Large-Scale Adaptive Electric Vehicle Charging.
            '''
            constraints_energy_delivered = []
            for active_ev in active_evs:
                row_index:int = infrastructure.get_station_index(active_ev.station_id)
                # probably need to specify indexes in better way
                # because it seems rates contain all-time rates
                r_t = cp.sum(rates[row_index, active_ev.arrival_offset: active_ev.arrival_offset + active_ev.remaining_time])



                # TODO: find why it is divided by 1e3
                predicted_energy_over_optimization_horizon = infrastructure.voltages[row_index] * r_t * delta_period / 1e3 / 60
                # constraints_energy_delivered.append(predicted_energy_over_optimization_horizon + active_ev.energy_delivered <= active_ev.requested_energy)

                constraints_energy_delivered.append(
                    predicted_energy_over_optimization_horizon  <= active_ev.remaining_demand)
            return constraints_energy_delivered
        def infrastructure_constraints():
            constraints_infrastructure = []

            if constraint_type == 'LINEAR':
                R = infrastructure.constraint_limits
                j = 0
                # what exactly v is?
                for v in infrastructure.constraint_matrix:
                    constraints_infrastructure.append(np.sum(np.abs(v) * rates) <= R[j])
                    j += 1
                return constraints_infrastructure
            elif constraint_type == 'SOC':
                phase_in_radians = np.deg2rad(infrastructure.phases)
                R = infrastructure.constraint_limits
                currents = []
                j = 0
                for v in infrastructure.constraint_matrix:
                    # maybe change currents to cvxpy array?
                    # currents = []
                    # currents.append(v*np.cos(phase_in_radians))
                    # currents.append(v*np.sin(phase_in_radians))
                    # currents = np.array(currents)
                    a = np.stack([v * np.cos(phase_in_radians), v * np.sin(phase_in_radians)])

                    # TODO: change to linalg norm if it is the same
                    #  (2,54) * (54,T) matrices
                    # constraints.append(cp.norm(currents @ rates) <= R[j])
                    constraints_infrastructure.append(cp.norm(a @ rates, axis=0) <= infrastructure.constraint_limits[j])
                    j += 1
                return constraints_infrastructure
            else:
                return ValueError('infrastructure_constraints->choose only LINEAR or SOC constraints!')

        # lower_charging_bounds, upper_charging_bounds = charging_rate_interval_constraints()
        # req_energy_constraints = requested_energy_charged_constraints()
        # infrastructure_consts = infrastructure_constraints()
        constraints = []
        return constraints
        # return (lower_charging_bounds,
        #         upper_charging_bounds,
        #         *req_energy_constraints,
        #         *infrastructure_consts)