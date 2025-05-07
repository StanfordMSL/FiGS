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
        hz:             Frequency of the controller.
        Nznn:           Number of states in the controller.
        nhy:            History sequence length.

    """
    def __init__(self,configs_path:Path=None) -> None:
        """
        Initialize the BaseController class.

        Args:
            configs_path: Path to the directory containing the JSON files.

        """
        # Set the configuration directory
        if configs_path is None:
            self.configs_path = Path(__file__).parent.parent.parent.parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Necessary attributes
        self.name:str = None
        self.hz:int = None
        self.zcr = None
        self.nhy:int = None

    @abstractmethod
    def set_initial_memory(self, x0: np.ndarray, u0:np.ndarray|None=None) -> None:
        """
        Set the initial memory of the controller.

        Args:
            x0: Initial state vector.
            u0: Initial control input vector (if any, None otherwise).

        """
        pass
    
    @abstractmethod
    def control(self, scr:dict[str,np.ndarray], zcr:dict[str,torch.Tensor]={}) -> tuple[np.ndarray, None|torch.Tensor, None|np.ndarray, np.ndarray]:
        """
        Abstract control method to be implemented by subclasses.

        Args:
            scr:    Dictionary containing the current sensor data.
            zcr:    Dictionary containing the current output feature vector (empty otherwise).

        Returns:
            ucr:    Control input.
            zcr:    Output feature vector (if any, None otherwise).
            aux:    Auxiliary outputs.
        """
        pass
