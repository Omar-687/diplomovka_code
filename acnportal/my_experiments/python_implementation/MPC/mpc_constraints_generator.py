import numpy as np
import cvxpy as cp
import numpy
from osqppurepy._osqp import info

from objective_enum import ObjectiveEnum
import warnings
from typing import List, Union, Optional
from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
from adacharge.adaptive_charging_optimization import AdaptiveChargingOptimization
from traitlets import Bool
class MPCConstraintsGenerator:
    def __init__(self):
        pass

    def charging_rate_interval_constraints(self,
                                           rates: cp.Variable,
                                           active_evs:List['SessionInfo'],
                                           infrastructure:InfrastructureInfo,
                                           optimisation_horizon):
        '''
        TODO: should i write description of method below or above params?
        # write it ABOVE, otherwise it will be in return section

        3a) constraint in the book Large-Scale Adaptive Electric Vehicle Charging.

        '''


        lower_charging_bounds = np.zeros(rates.shape)
        upper_charging_bounds = np.zeros(rates.shape)

        for active_ev in active_evs:
            row_index: int = infrastructure.get_station_index(station_id=active_ev.station_id)

            # TODO: check if max and min rate should be taken from evse class or from ev class

            max_rates = active_ev.max_rates[0: len(optimisation_horizon) - 1]

            # optimisation horizon can be longer than remaining time, therefore we use minimum
            # the rest is filled with zeros to satisfy 0 <= r <= 0 (r=0) from 3b)
            # active_ev_remaining_time == len(max_rates)
            upper_charging_bounds[row_index][0: min(len(optimisation_horizon) - 1,
                                                    active_ev.remaining_time)] = max_rates


        return lower_charging_bounds <= rates, rates <= upper_charging_bounds

    def convert_pilot_to_charging_rate(self, pilot,
                                       infrastructure,
                                       evse_index,
                                       period_between_timesteps ):
        period_between_timesteps_in_minutes = period_between_timesteps
        watts_per_min = infrastructure.voltages[evse_index] * pilot * period_between_timesteps_in_minutes
        kilowatts_per_min = watts_per_min / 1000
        kilowatts_per_hour = kilowatts_per_min / 60
        return kilowatts_per_hour

    def requested_energy_charged_constraints(self,
                                             rates: cp.Variable,
                                             active_evs:List['SessionInfo'],
                                             infrastructure:InfrastructureInfo,
                                             optimisation_horizon:list,
                                             period_between_timesteps:float):
        '''
        3b) constraint in the publication Large-Scale Adaptive Electric Vehicle Charging.
        '''
        constraints_energy_delivered = []
        for active_ev in active_evs:
            evse_index: int = infrastructure.get_station_index(active_ev.station_id)

            r_t = cp.sum(rates[evse_index, 0:min(len(optimisation_horizon) - 1,
                                                 active_ev.remaining_time)])
            predicted_energy_over_optimization_horizon = (
                self.convert_pilot_to_charging_rate(pilot=r_t,
                                                    infrastructure=infrastructure,
                                                    evse_index=evse_index,
                                                    period_between_timesteps=period_between_timesteps))
            constraints_energy_delivered.append(active_ev.energy_delivered +
                                                predicted_energy_over_optimization_horizon
                                                <= active_ev.requested_energy)
        return constraints_energy_delivered


    def infrastructure_constraints(self,
                                   rates,
                                   constraint_type,
                                   infrastructure):
        constraints_infrastructure = []

        # test if correct
        if constraint_type == 'LINEAR':
            R = infrastructure.constraint_limits
            # what exactly v is?
            for j in range(len(R)):
                charging_rates = infrastructure.constraint_matrix[j] @ rates
                aggregate_charging_rate = cp.norm(cp.sum(charging_rates), axis=0)
                constraints_infrastructure.append(aggregate_charging_rate <= R[j])

            return constraints_infrastructure

        elif constraint_type == 'SOC':
            evse_phase_in_radians = np.deg2rad(infrastructure.phases)
            R = infrastructure.constraint_limits
            for j in range(len(R)):
                # maybe change currents to cvxpy array?
                arr = np.stack([infrastructure.constraint_matrix[j] * np.cos(evse_phase_in_radians),
                                infrastructure.constraint_matrix[j] * np.sin(evse_phase_in_radians)])

                # constraints.append(cp.norm(currents @ rates) <= R[j])
                constraints_infrastructure.append(cp.norm(arr @ rates, axis=0) <= R[j])

            return constraints_infrastructure
        else:
            return ValueError('infrastructure_constraints->choose only LINEAR or SOC constraints!')

    def get_all_constraints(self,
                            rates:cp.Variable,
                            active_evs: List["SessionInfo"],
                            infrastructure: InfrastructureInfo,
                            optimisation_horizon:List[int],
                            constraint_type:str,
                            period_between_timesteps:float,
                            # minimal_value_charging=False
                            ):

        lower_charging_bounds, upper_charging_bounds = self.charging_rate_interval_constraints(rates=rates,
                                                                                               active_evs=active_evs,
                                                                                               infrastructure=infrastructure,
                                                                                               optimisation_horizon=optimisation_horizon)
        req_energy_constraints = self.requested_energy_charged_constraints(rates=rates,
                                                                           active_evs=active_evs,
                                                                           infrastructure=infrastructure,
                                                                           optimisation_horizon=optimisation_horizon,
                                                                           period_between_timesteps=period_between_timesteps)
        infrastructure_consts = self.infrastructure_constraints(rates=rates,
                                                                constraint_type=constraint_type,
                                                                infrastructure=infrastructure)
        return (lower_charging_bounds,
                upper_charging_bounds,
                *req_energy_constraints,
                *infrastructure_consts)