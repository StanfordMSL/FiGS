# Developed from: https://github.com/madang6/flightroom_ns_process/tree/feature/video_process

from pathlib import Path
import json
from typing import List, Tuple, Dict, Union

from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)

import figs.utilities.capture_helper as ch
import cv2
import numpy as np
import open3d as o3d
import subprocess

def generate_gsplat(scene_file_name:str,capture_cfg_name:str='default',
                    gsplats_path:Path=None,config_path:Path=None) -> None:
    
    # Initialize base paths
    if gsplats_path is None:
        gsplats_path = Path(__file__).parent.parent.parent.parent.parent/'gsplats'

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent.parent/'configs'

    capture_cfg_path = config_path/'captures'
    capture_path = gsplats_path/'capture'
    workspace_path = gsplats_path/'workspace'
    
    # Find the correct video path
    video_files = list(capture_path.glob(f"*{scene_file_name}*"))
    if len(video_files) == 0:
        raise FileNotFoundError(f"No file found with name containing '{scene_file_name}' in {capture_path}")
    elif len(video_files) > 1:
        raise ValueError(f"Multiple files found with name containing '{scene_file_name}' in {capture_path}")
    else:
        video_path = str(video_files[0])

    # Initialize process paths
    process_path = workspace_path / scene_file_name

    images_path = process_path / "images"
    spc_path = process_path / "sparse_pc.ply"
    tfm_path = process_path / "transforms.json"

    sfm_path = process_path / "sfm"
    sfm_spc_path = sfm_path / "sparse_pc.ply"
    sfm_tfm_path = sfm_path / "transforms.json"
    
    process_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    # Initialize output paths
    outputs_path = workspace_path/'outputs'
    output_path = outputs_path/scene_file_name

    output_path.mkdir(parents=True, exist_ok=True)

    # Load the capture config
    capture_config_file = capture_cfg_path/f"{capture_cfg_name}.json"
    with open(capture_config_file, "r") as file:
        capture_configs = json.load(file)

    camera_config = capture_configs["camera"]
    extractor_config = capture_configs["extractor"]

    # Extract the frame data
    extract_frames(video_path,images_path,extractor_config)
    
    # Run the ns_process step
    ns_obj = ImagesToNerfstudioDataset(
        data=images_path, output_dir=sfm_path,
        camera_type="perspective", matching_method="exhaustive",sfm_tool="hloc",gpu=True
    )
    ns_obj.main()

    # Load the resulting transforms.json and sparse_points.ply
    with open(sfm_tfm_path, "r") as f:
        tfm_data = json.load(f)
    
    sparse_pcloud = o3d.io.read_point_cloud(sfm_spc_path.as_posix())

    # Use sfm config if camera config is not provided
    if camera_config is None:
        fx,fy = tfm_data["fl_x"],tfm_data["fl_y"]
        cx,cy = tfm_data["cx"],tfm_data["cy"]
        k1,k2 = tfm_data["k1"],tfm_data["k2"]
        p1,p2 = tfm_data["p1"],tfm_data["p2"]

        camera_config = {
            "model": tfm_data["camera_model"],
            "height": tfm_data["h"],
            "width": tfm_data["w"],
            "intrinsics_matrix": [
                [ fx, 0.0,  cx],
                [0.0,  fy,  cy],
                [0.0, 0.0, 1.0]
            ],
            "distortion_coefficients": [k1,k2,p1,p2]
        }
        
    # Compute the transform using aruco markers
    Psfm,Parc = extract_positions(sfm_path,extractor_config,camera_config)
    cs,Rs,ts = ch.compute_ransac_transform(Psfm,Parc)

    # Generate the sparse point cloud and transform files
    for frame in tfm_data["frames"]:
        Tc2s = np.array(frame["transform_matrix"])

        Tc2w = np.eye(4)
        Tc2w[:3,:3],Tc2w[:3,3] = Rs@Tc2s[:3,:3],cs*Rs@Tc2s[:3,3] + ts

        frame["transform_matrix"] = Tc2w.tolist()

    sparse_points = np.asarray(sparse_pcloud.points)
    for idx, point in enumerate(sparse_points):
        sparse_points[idx,:] = cs*Rs@point + ts

    sparse_pcloud.points = o3d.utility.Vector3dVector(sparse_points)

    # Save the updated files
    with open(tfm_path, "w", encoding="utf8") as f:
        json.dump(tfm_data, f, indent=4)

    o3d.io.write_point_cloud(spc_path.as_posix(),sparse_pcloud)

    # Run the gsplat generation
    command = [
        "ns-train",
        "splatfacto",
        "--data", scene_file_name,
        "--viewer.quit-on-train-completion", "True",
        "--output-dir", 'outputs',
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
        "nerfstudio-data",
        "--orientation-method", "none",
        "--center-method", "none",
        "--auto-scale-poses", "False"
    ]

    # Run the command
    result = subprocess.run(command, cwd=workspace_path.as_posix(), capture_output=True, text=True)

    # Check the result
    if result.returncode == 0:
        print("Command succeeded.")
        print(result.stdout)  # Output of the command
    else:
        print("Command failed.")
        print(result.stderr)  # Error output
    
def extract_frames(video_path:Path,rgbs_path:Path,
                   extractor_config:Dict['str',Union[int,float]]) -> List[np.ndarray]:
    """
    Extracts frame data from video into a folder of images.
    
    """

    # Unpack the extractor configs
    Nimg = extractor_config["num_images"]
    Narc = extractor_config["num_marked"]
    mkr_id = extractor_config["marker_id"]

    # Initialize the aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Cannot open the video file.")
    
    # Survey frames for aruco markers
    Ntot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Tarc,Temp = [],[]
    for _ in range(Ntot):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame has an aruco marker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, ids, _ = detector.detectMarkers(gray)

        # Bin the frame by the marker detection
        if ids is not None and len(ids) == 1 and ids[0] == mkr_id:
            Tarc.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            Temp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Check if enough aruco markers were found
    if len(Tarc) < Narc:
        Tout = Tarc + ch.distribute_values(Temp,Nimg-len(Tarc))
        print(f"Warning: Only {len(Tarc)} aruco markers found. Using {Narc-len(Tarc)} empty frames to fill the gap.")
    else:
        Tout = ch.distribute_values(Tarc,Narc) + ch.distribute_values(Temp,Nimg-Narc)
    
    Tout.sort()

    # Extract the selected frames
    for idx, tout in enumerate(Tout):
        cap.set(cv2.CAP_PROP_POS_MSEC,tout)
        ret, frame = cap.read()
        if not ret:
            break

        # Save the image
        rgb_path = rgbs_path / f"frame_{idx+1:05d}.png"
        cv2.imwrite(str(rgb_path),frame)

    # Release the video capture object
    cap.release()

def extract_positions(sfm_path:Path,
                      extractor_config:Dict['str',Union[int,float]],
                      camera_config:Dict['str',Union[int,float]]=None) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    # Unpack the extractor configs
    Narc = extractor_config["num_marked"]
    marker_length = extractor_config["marker_length"]
    marker_id = extractor_config["marker_id"]

    # Unpack the camera configs
    if camera_config is None:
        # TODO: Add option to use SfM camera parameters
        raise ValueError("Camera configuration is not provided.")
    else:
        camera_matrix = np.array(camera_config["intrinsics_matrix"])
        dist_coeffs = np.array(camera_config["distortion_coefficients"])

    # Initialize the aruco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ])
    
    # Open the transforms.json file
    with open(sfm_path / "transforms.json", "r") as f:
        transforms = json.load(f)
    frames = transforms["frames"]

    TTarc,TTsfm = [],[]
    for frame in frames:
        # Open the image file
        image_path = sfm_path.parent / frame["file_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Error: Cannot open the image file {image_path}")
        
        # Detect the aruco marker in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 1 and ids[0] == marker_id:            
            # Compute the Aruco transform
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            # Compute the transforms
            if ret:
                Tw2c_arc = np.eye(4)
                Tw2c_arc[:3, :3],Tw2c_arc[:3, 3] = cv2.Rodrigues(rvec)[0],tvec.flatten()  # world to camera

                # Compute the camera to world transforms
                Tarc = np.linalg.inv(Tw2c_arc)
                Tsfm = np.array(frame["transform_matrix"])

                TTarc.append(Tarc)
                TTsfm.append(Tsfm)
    
    # Check if the number of transforms match our expectations
    if len(TTarc) != Narc:
        raise ValueError("Error: Mismatched number of aruco and sfm transforms.")

    # Extract the positions
    Parc = np.array([T[:3,3] for T in TTarc]).T
    Psfm = np.array([T[:3,3] for T in TTsfm]).T

    return Psfm,Parc