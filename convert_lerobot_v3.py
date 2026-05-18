import os
from typing import List

from numpy import ndarray
import numpy

import lerobot
assert lerobot.__version__ == "0.5.1", f"Error: this script requires lerobot v0.5.1 but v{lerobot.__version__} was found instead"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rlbench_dataset_utils.observation_config import CameraConfig, ObservationConfig
import rlbench_dataset_utils.utils 

# DATASETS PATHS
RLBENCH_DATASET_ROOT = "./datasets/rlbench/generated-20ep"
TASKS = os.listdir(RLBENCH_DATASET_ROOT)
LEROBOT_REPO_ID_V3 = "SimonReese/lerobot-20-ep-v3"
LEROBOT_DATASET_ROOT_V3 = "./datasets/lerobot/v3/lerobot-20-ep-v3"

# CAMERA CONFIGURATIONS
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
CAMERA_CONFIG = CameraConfig(
    image_size= IMAGE_SIZE
)
OBS_CONFIG = ObservationConfig(
    left_shoulder_camera= CAMERA_CONFIG,
    right_shoulder_camera= CAMERA_CONFIG,
    overhead_camera= CAMERA_CONFIG,
    wrist_camera= CAMERA_CONFIG,
    front_camera= CAMERA_CONFIG,
    gripper_joint_positions=True
)

# LeRobot features dictionary
FEATURES_DICT = {
    "image": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["width", "height", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["width", "height", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (8,), # Joint states (7) + Gripper open amount (1 is open, 0 is closed)
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,), # Delta gripper pose action (xyz rx ry rz)  + next gripper open amout
        "names": ["actions"],
    }
}

FRAME_DICT = {
    "image": ndarray,
    "wrist_image": ndarray,
    "state" : ndarray,
    "actions" : ndarray,
    "task": str
}

def main():

    lerobot_dataset = LeRobotDataset.create(
        repo_id=LEROBOT_REPO_ID_V3,
        fps=10,
        features=FEATURES_DICT,
        root=LEROBOT_DATASET_ROOT_V3,
        robot_type="panda"
    )

    # Open every task
    for TASK in TASKS:
        VARIATIONS = get_variations_ids(RLBENCH_DATASET_ROOT, TASK)
        # Open every variation
        for VARIATION in VARIATIONS:
            EPISODES = get_episodes_number(RLBENCH_DATASET_ROOT, TASK, VARIATION)
            # Open every episode
            for EPISODE in EPISODES:
                
                # Get the demo for the episode, since loading all episodes is expensive
                DEMOS = rlbench_dataset_utils.utils.get_stored_demos(
                    amount=1,
                    image_paths=False,
                    dataset_root=RLBENCH_DATASET_ROOT,
                    variation_number=VARIATION,
                    task_name=TASK,
                    obs_config=OBS_CONFIG,
                    random_selection=False,
                    from_episode_number=EPISODE
                )
                DEMO = DEMOS.pop()
                print(f"Processing TASK:{TASK}, VARIATION:{VARIATION}, EP:{EPISODE}:\n{DEMO.demo_description}")
                # Given an image at timestep t, we want the action to reach next position
                for seq, observation in enumerate(DEMO):
                    # Get the next observation
                    if seq+1 >= len(DEMO): break # No more obs available
                    next_obs = DEMO[seq+1]
                    current_pose = observation.gripper_pose
                    next_pose = next_obs.gripper_pose
                    # Compute delta pose
                    delta_pose = rlbench_dataset_utils.utils.delta_pose_ee(current_pose[:3], current_pose[3:], next_pose[:3], next_pose[3:])
                    # Convert quaternion rotation to euler
                    rotation_xyz = rlbench_dataset_utils.utils.quaternion_to_euler(delta_pose[3:])
                    # Get next opening amount of gripper
                    next_open_amount = rlbench_dataset_utils.utils.get_panda_gripper_open_amount(next_obs.gripper_joint_positions)[0]
                    # Store action
                    FRAME_DICT["actions"] = numpy.concatenate((delta_pose[:3], rotation_xyz, [next_open_amount]),dtype=numpy.float32) # TODO: convert to delta xyz

                    # Store images
                    FRAME_DICT["image"] = observation.front_rgb # TODO: Check image format and shape
                    FRAME_DICT["wrist_image"] = observation.wrist_rgb
                    
                    # Store states
                    joint_states = observation.joint_positions
                    gripper_amount = rlbench_dataset_utils.utils.get_panda_gripper_open_amount(observation.gripper_joint_positions)[0]
                    FRAME_DICT["state"] = numpy.concatenate((joint_states, [gripper_amount]), dtype=numpy.float32)  
                   
                    # Store task description
                    if type(DEMO.demo_description) == list:
                        FRAME_DICT["task"] = DEMO.demo_description[0]
                    else:
                        FRAME_DICT["task"] = DEMO.demo_description

                    lerobot_dataset.add_frame(FRAME_DICT)
                lerobot_dataset.save_episode()
                
    lerobot_dataset.finalize()
        

# ----- UTILITY FUNCTIONS -----

# Get list of variations for task
def get_variations_ids(dataset_path: str, task_name:str, VARIATION_FOLDER_PREFIX = "variation") -> List[int]:
    # Open variation
    VARIATION_FOLDER_PREFIX = "variation"
    variation_folders = os.listdir(os.path.join(dataset_path, task_name))
    if "all_variations" in variation_folders: variation_folders.remove("all_variations")
    variation_ids = []
    for variation in variation_folders:
        if not os.path.isdir(os.path.join(dataset_path, task_name, variation)): continue
        if VARIATION_FOLDER_PREFIX not in variation: continue
        id = variation.removeprefix(VARIATION_FOLDER_PREFIX)
        variation_ids.append(int(id))
    return variation_ids

def get_episodes_number(dataset_path: str, task_name:str, variation_id: int, VARIATION_FOLDER_PREFIX = "variation", EPISODES_FOLDER = "episodes",EPISODE_FOLDER_PREFIX = "episode") -> List[int]:
    episodes_folders = os.listdir(os.path.join(dataset_path, task_name, f"{VARIATION_FOLDER_PREFIX}{variation_id}", EPISODES_FOLDER))
    episode_ids = []
    for ep in episodes_folders:
        if not os.path.isdir(os.path.join(dataset_path, task_name, f"{VARIATION_FOLDER_PREFIX}{variation_id}", EPISODES_FOLDER, ep)): continue
        if EPISODE_FOLDER_PREFIX not in ep: continue
        id = ep.removeprefix(EPISODE_FOLDER_PREFIX)
        episode_ids.append(int(id))
    episode_ids.sort()
    return episode_ids

if __name__ == "__main__":
    main()