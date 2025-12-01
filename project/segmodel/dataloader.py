"""
Mapping Dataset:
Carla: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
TrafficCAM: Truck, Bus, Motor Bike, bike, Pedestrian, LMV

Not in Carla (NC): Tractor(NC), E-rickshaw(NC), LCV(NC), Auto(NC)  

Carla to TrafficCam (the labels we will save):
Truck --> Truck
Bus --> Bus
Motorcycle --> Motor Bike
Bicycle --> bike
Pedestrian --> Pedestrian
Car --> LMV
"""
import torch
import os
import json
import numpy as np
from enum import Enum
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Label(Enum):
    IGNORE = 255
    TRUCK = 1
    BUS = 2
    MOTORCYCLE = 3
    BICYCLE = 4
    PEDESTRIAN = 5
    CAR = 6

# RGB to Label
CARLA_TO_LABEL = {
    (0, 0, 70): Label.TRUCK,
    (0, 60, 100): Label.BUS,
    (0, 0, 230): Label.MOTORCYCLE,
    (119, 11, 32): Label.BICYCLE,
    (220, 20, 60): Label.PEDESTRIAN,
    (0, 0, 142): Label.CAR
}

# string to label
TRAFFIC_CAM_TO_LABEL = {
    "Truck": Label.TRUCK,
    "Bus": Label.BUS,
    "MotorBike": Label.MOTORCYCLE,
    "Bike": Label.BICYCLE,
    "Pedestrian": Label.PEDESTRIAN,
    "LMV": Label.CAR
}

class CarlaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        sort_key = lambda file: int(file[len("frame_"):-1*len(".png")])
        self.images = sorted(os.listdir(image_dir), key=sort_key)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"[WARNING] Corrupt image skipped: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        mask_rgb = Image.open(mask_path).convert("RGB")
        mask = self.rgb_to_mask(np.array(mask_rgb))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.

        return image.float(), torch.from_numpy(mask).long()

    def rgb_to_mask(self, rgb):
        h, w, _ = rgb.shape
        mask = np.full((h, w), Label.IGNORE.value, dtype=np.uint8)

        for color, label in CARLA_TO_LABEL.items():
            matches = np.all(rgb == color, axis=-1)
            mask[matches] = label.value

        return mask