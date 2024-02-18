from typing import List, Union, Optional
from adacharge import AdaptiveChargingOptimization
import numpy as np
from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo
class MPCOptimizer(AdaptiveChargingOptimization):
    def __init__(self, interface, objective):
        super().__init__(interface, objective)

