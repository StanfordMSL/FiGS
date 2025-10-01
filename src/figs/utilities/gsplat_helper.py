"""
Helper functions for loading configs.
"""
import os

from pathlib import Path
from figs.render.gsplat import GSplat

def get_gsplat(scene_name:str, gsplats_path:Path=None):
    """"
    Load a configuration file from the corresponding configs directory.

    Args:
        - scene_name: Name of the gsplat file.
        - gsplats_path: Path to the gsplats directory.

    Returns:
        - gsplat: GSplat object.

    """
    # Set the gsplats directory if not provided
    if gsplats_path is None:
        gsplats_path = Path(__file__).parent.parent.parent.parent.parent/'gsplats'

    curr_path = Path.cwd()
    wspace_path = gsplats_path/'workspace'
    search_path = wspace_path/'outputs'/scene_name
    
    # Find the GSplat configuration
    yaml_configs = list(search_path.rglob("*.yml"))
    
    if len(yaml_configs) == 0:
        raise ValueError(f"The search path '{search_path}' did not return any configurations.")
    elif len(yaml_configs) > 1:
        raise ValueError(f"The search path '{search_path}' returned multiple configurations. Please specify a unique configuration within the directory.")
    else:
        gsplat_config = yaml_configs[0]

    # Load GSplat (from the workspace directory to avoid path issues)
    os.chdir(wspace_path)
    gsplat = GSplat(gsplat_config)
    os.chdir(curr_path)

    return gsplat
