import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from improved_training import MaskDataset, SegmentationHeadConvNeXt, compute_iou, n_classes
import torchvision.transforms as transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "C:/Users/lucky/OneDrive/Desktop/dualityai/Offroad_Segmentation_Training_Dataset/val"

w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((h, w), interpolation=Image.NEAREST),
])

dataset = MaskDataset(data_dir, transform, mask_transform)
loader = DataLoader(dataset, batch_size=2)

backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(device)

imgs, _ = next(iter(loader))
imgs = imgs.to(device)

with torch.no_grad():
    features = backbone.forward_features(imgs)["x_norm_patchtokens"]

n_embedding = features.shape[2]

model = SegmentationHeadConvNeXt(
    in_channels=n_embedding,
    out_channels=n_classes,
    tokenW=w // 14,
    tokenH=h // 14
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

iou_scores = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        features = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(features)
        outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
        iou_scores.append(compute_iou(outputs, labels))

print("Mean IoU:", np.mean(iou_scores))