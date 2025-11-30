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
from enum import Enum

class Label(Enum):
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

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        pass

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pass