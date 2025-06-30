import numpy as np

class ExternalForces:
    """
    Class to handle external forces acting on the quadcopter.
    
    Args:
        - forces_config: Dict of external forces (e.g., wind, thrust).
    """
    
    def __init__(self, forces_config: dict|None) -> None:
        self.sources = {}

        if forces_config is None:
            forces_config = {}

        for name,config in forces_config.items():
            fext = {
                "lower": np.array(config["lower"]),
                "upper": np.array(config["upper"]),
                "mean": np.array(config["mean"]),
                "std": np.array(config["std"])
            }

            self.sources[name] = fext

    def get_forces(self, pv_cr: np.ndarray, noisy:bool=False) -> np.ndarray:
        """
        Apply external forces to the quadcopter state.
        
        Args:
            - pv_cr: Current position and velocity of the quadcopter.
            - noisy: Boolean to indicate if noise should be added.
        
        Returns:
            - forces: External forces applied to the quadcopter.
        """
        forces = np.zeros(3)
        for source in self.sources.values():
            if np.all((pv_cr[0:3] <= source["upper"]) & (pv_cr[0:3] >= source["lower"])):
                force = source["mean"]
                if noisy:
                    force += np.random.normal(0, source["std"])

                forces += force

        return forces