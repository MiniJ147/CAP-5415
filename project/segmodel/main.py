import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader 

import dataloader
from models import DDRNet_23 as DDRNet

def mean_iou(pred, target, num_classes=7):
    pred = pred.view(-1)
    target = target.view(-1)

    valid = target != dataloader.Label.IGNORE.value
    pred = pred[valid]
    target = target[valid]

    ious = []

    for cls in range(1, num_classes):  # skip ignore and unused 0
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if len(ious) else 0

# config
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(DEVICE)

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])


dataset = dataloader.CarlaDataset(
    image_dir="data/carla_captures/Foggy/camera0/raw",
    mask_dir="data/carla_captures/Foggy/camera0/semantic",
    transform=transform,
)

train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
)

model = DDRNet.get_seg_model(False, num_classes=7).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=dataloader.Label.IGNORE.value)

optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    running_iou = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, masks in pbar:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        iou = mean_iou(preds, masks)

        running_loss += loss.item()
        running_iou += iou

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "IoU": f"{iou:.3f}"
        })

    print(f"Epoch {epoch+1}: "
          f"Loss={running_loss/len(train_loader):.4f}, "
          f"IoU={running_iou/len(train_loader):.3f}")