import numpy as np
import torch
import json

from pathlib import Path
from abc import ABC, abstractmethod

class BaseController(ABC):
    """
    Abstract base class for controllers.

    Methods:
        control(**inputs): Abstract method to be implemented by subclasses.
    
    Attributes:
        configs_path:   Path to the directory containing the JSON files.
        name:           Name of the controller.
        hz:             Frequency of the controller.
        Nhy:            History sequence length.
        Nhn:            Horizon sequence length.
    """
    def __init__(self,configs_path:Path=None) -> None:
        """
        Initialize the BaseController class.

        Args:
            configs_path: Path to the directory containing the JSON files.

        """

        # Necessary attributes
        self.name:str = None                # Name of the controller
        self.hz:int = None                  # Frequency of the controller
    
    @abstractmethod
    def control(self,tcr:float,xcr:np.ndarray,upr:np.ndarray,
                rgb:np.ndarray,dpt:np.ndarray,
                fcr:np.ndarray
    ) -> tuple[np.ndarray, dict[str,np.ndarray|torch.Tensor]]:
        """
        Abstract control method to be implemented by subclasses.

        Args:
            tcr: Current time.
            xcr: Current state.
            upr: Previous control input.
            rgb: RGB image.
            dpt: Depth image.
            fcr: Current force.

        Returns:
            ucr: Controller output.
            aux: Auxiliary outputs.
        """
        pass
