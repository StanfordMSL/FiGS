import numpy as np
import torch
import json

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict,Union,Tuple,List

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
        self.Nznn:Dict[str,List[torch.Tensor]] = None
        self.nhy:int = None

    @abstractmethod
    def control(self, tcr: float, xcr: np.ndarray,
                upr: Union[None, np.ndarray],
                obj: Union[None, np.ndarray],
                icr: Union[None, np.ndarray],
                zcr: Union[None, torch.Tensor]
                ) -> Tuple[np.ndarray, Union[None, torch.Tensor], Union[None, np.ndarray], np.ndarray]:
        """
        Abstract control method to be implemented by subclasses.

        Args:
            tcr: Time at the current control step.
            xcr: States at the current control step.
            upr: Previous control step inputs (if any, None otherwise).
            obj: Objective vector (if any, None otherwise).
            icr: Image at the current control step (if any, None otherwise).
            zcr: Feature vector at current control step (if any, None otherwise).

        Returns:
            ucr: Control input.
            zcr: Output feature vector (if any, None otherwise).
            adv: Advisor output (if any, None otherwise).
            tsol: Time taken to solve components.

        """
        pass

    def load_json_config(self, config:str, name:str) -> Dict:
        """
        Load a JSON configuration file.

        Args:
            config: Name of the configuration directory.
            name: Name of the configuration file.

        Returns:
            config: Dictionary containing the configuration.
        
        """
        json_config = self.configs_path/config/(name+".json")

        if not json_config.exists():
            raise ValueError(f"The json file '{json_config}' does not exist.")
        else:
            # Load the json configuration
            with open(json_config) as file:
                config = json.load(file)
            
            # Add the name to the configuration
            config["name"] = name

        return config