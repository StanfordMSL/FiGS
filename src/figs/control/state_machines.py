import time
import shutil
import os
import numpy as np
from typing import Literal
from enum import Enum

import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th
import figs.dynamics.quadcopter_rate_model as qrm

from casadi import Function
from acados_template import AcadosOcp, AcadosOcpSolver
from figs.control.base_controller import BaseController
from figs.dynamics.external_forces import ExternalForces
from figs.visualize import rich_visuals as rv

class bMode(Enum):
    RESET = 0
    APPROACH = 1
    CONTACT = 2

def check_conditions( conditions:dict[str, float|None], tolerances:dict[str,float|None]):
    """
    Method to check if the current conditions meet the specified tolerances.

    Args:
        - conditions: Current conditions dictionary.
        - tolerances: Tolerances dictionary.
    """

    for key,val in tolerances.items():
        if val is None:
            continue  # automatically true
        else:
            if isinstance(val, list):
                if val[0] == "leq" and val[1] < conditions[key]:
                    return False
                elif val[0] == "geq" and val[1] > conditions[key]:
                    return False
    return True