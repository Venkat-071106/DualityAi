import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from improved_training import MaskDataset, SegmentationHeadConvNeXt, n_classes
import torchvision.transforms as transforms
from PIL import Image

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

class_iou = np.zeros(n_classes)
counts = np.zeros(n_classes)

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        features = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(features)
        preds = torch.argmax(logits, dim=1)

        for cls in range(n_classes):
            pred_mask = preds == cls
            true_mask = labels == cls

            intersection = (pred_mask & true_mask).sum().item()
            union = (pred_mask | true_mask).sum().item()

            if union > 0:
                class_iou[cls] += intersection / union
                counts[cls] += 1

class_iou = class_iou / (counts + 1e-6)

df = pd.DataFrame({
    "Class_ID": list(range(n_classes)),
    "IoU": class_iou
})

df.to_csv("results/per_class_iou.csv", index=False)

print("Per-class IoU saved to results/per_class_iou.csv")