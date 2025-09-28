"""
Helper functions for loading configs.
"""
import json

from pathlib import Path

def get_config(config_name:str, config_type:str,configs_path:Path=None) -> dict: 
    """"
    Load a configuration file from the corresponding configs directory.

    Args:
        - config_name: Name of the configuration file.
        - config_type: Type of configuration file.
        - configs_path: Path to the configs directory.

    Returns:
        - config: Configuration dictionary.
    """

    # Set the configurations directory if not provided
    if configs_path is None:
        configs_path = Path(__file__).parent.parent.parent.parent.parent/'configs'

    # Load the config
    config_path = configs_path/config_type/(config_name+".json")

    if config_path.exists():
        with open(config_path) as file:
            config = json.load(file)
    else:
        raise ValueError(f"The json file '{config_path}' does not exist.")
        
    return config