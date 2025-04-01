import numpy as np
import torch

from pathlib import Path
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from typing import Dict, Union 

class GSplat():
    def __init__(self, gsplat_path:Path) -> None:
        """
        GSplat class for rendering images from GSplat pipeline.

        Args:
            - gsplat_config: FiGS gsplat configuration dictionary.

        Variables:
            - device: Device to run the pipeline on.
            - config: Configuration for the pipeline.
            - pipeline: Pipeline for the GSplat model.
            - T_w2g: Transformation matrix from world to GSplat frame.

        """

        # Some useful intermediate variables
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Tw2g = np.array([
            [ 1.00, 0.00, 0.00, 0.00],
            [ 0.00,-1.00, 0.00, 0.00],
            [ 0.00, 0.00,-1.00, 0.00],
            [ 0.00, 0.00, 0.00, 1.00]
        ])

        # Class variables
        self.device = device
        self.config,self.pipeline, _, _ = eval_setup(gsplat_path,test_mode="inference")
        self.Tw2g = Tw2g

    def generate_output_camera(self, camera_config:Dict[str,Union[int,float]]) -> Cameras:
        """
        Generate an output camera for the pipeline.

        Args:
            - camera_config: Configuration dictionary for the camera. Contains
                             width, height, channels, fx, fy, cx, cy.

        Returns:
            - camera_out: Output camera for the pipeline.
            
        """
        # Extract the camera parameters
        width,height = camera_config["width"],camera_config["height"]
        fx,fy = camera_config["fx"],camera_config["fy"]
        cx,cy = camera_config["cx"],camera_config["cy"]

        # Create the camera object
        camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]
        camera_out = Cameras(
            camera_to_worlds=1.0*camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out
            
    def render_rgb(self, camera:Cameras,T_c2w:np.ndarray) -> np.ndarray:
        """
        Render an RGB image from the GSplat pipeline.

        Args:
            - camera: Camera object for the pipeline.
            - T_c2w: Transformation matrix from camera to world frame.

        Returns:
            - image_rgb: Rendered RGB image.

        """

        # Extract the camera to gsplat pose
        Tc2g = self.Tw2g@T_c2w
        Pc2g = torch.tensor(Tc2g[0:3,:]).float()

        # Render rgb image from the pose
        camera.camera_to_worlds = Pc2g[None,:3, ...]
        with torch.no_grad():
            image_rgb = self.pipeline.model.get_outputs_for_camera(camera, obb_box=None)["rgb"]

        # Convert to output image
        image_rgb = image_rgb.cpu().numpy()             # Convert to numpy
        image_rgb = (255*image_rgb).astype(np.uint8)    # Convert to uint8

        return image_rgb