from PIL import Image
from lerobot.datasets import LeRobotDataset
from sympy import im
import torch
from torchvision.transforms.v2 import ToPILImage
# DATASETS PATHS
LEROBOT_REPO_ID_V3 = "SimonReese/lerobot-20-ep-v3"
LEROBOT_DATASET_ROOT_V3 = "./datasets/lerobot/v3/lerobot-20-ep-v3"
# LeRobot features dictionary
# FEATURES_DICT = {
#     "image": {
#         "dtype": "image",
#         "shape": IMAGE_SHAPE,
#         "names": ["width", "height", "channel"],
#     },
#     "wrist_image": {
#         "dtype": "image",
#         "shape": IMAGE_SHAPE,
#         "names": ["width", "height", "channel"],
#     },
#     "state": {
#         "dtype": "float32",
#         "shape": (8,), # Joint states (7) + Gripper open amount (1 is open, 0 is closed)
#         "names": ["state"],
#     },
#     "actions": {
#         "dtype": "float32",
#         "shape": (7,), # Delta gripper pose action (xyz rx ry rz)  + next gripper open amout
#         "names": ["actions"],
#     }
# }

dataset = LeRobotDataset(
    repo_id=LEROBOT_REPO_ID_V3,
    root=LEROBOT_DATASET_ROOT_V3,
)
print(f"Dataset: {dataset.repo_id}\n- num_ep: {dataset.num_episodes}")
print(dataset[0].keys())
frame: dict
idx = 0
for frame in dataset:
    if idx == 10: break
    img_tensor: torch.Tensor = frame["image"]
    conv = ToPILImage()
    img = conv(img_tensor)
    #img.save("image.png")
    print(f"Performing action: {frame["actions"]}")
    idx += 1
