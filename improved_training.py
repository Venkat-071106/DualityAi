"""
Improved Segmentation Training Script
Fine-tuning version (Resume + Cosine + AMP + Best Model Saving)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm


# ============================================================
# Mask Mapping
# ============================================================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================
# Dataset
# ============================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform

        image_files = set(os.listdir(self.image_dir))
        mask_files = set(os.listdir(self.masks_dir))

        self.data_ids = list(image_files & mask_files)
        print(f"Matched image-mask pairs: {len(self.data_ids)}")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]

        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


# ============================================================
# Model Head
# ============================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================
# Metrics
# ============================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []

    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


# ============================================================
# Main Training
# ============================================================

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Fine-tuning settings
    batch_size = 2
    lr = 2e-5
    n_epochs = 35

    # Windows local dataset path
    data_dir = "C:/Users/lucky/OneDrive/Desktop/dualityai/Offroad_Segmentation_Training_Dataset/train"
    val_dir  = "C:/Users/lucky/OneDrive/Desktop/dualityai/Offroad_Segmentation_Training_Dataset/val"

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
    ])

    trainset = MaskDataset(data_dir, transform, mask_transform)
    valset = MaskDataset(val_dir, transform, mask_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        output = backbone.forward_features(imgs)["x_norm_patchtokens"]

    n_embedding = output.shape[2]

    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    # Resume previous model if exists
    if os.path.exists("segmentation_head.pth"):
        classifier.load_state_dict(torch.load("segmentation_head.pth", map_location=device))
        print("Previous model loaded successfully!")

    best_iou = 0
    print("\nStarting Fine-Tuning...\n")

    for epoch in range(n_epochs):

        classifier.train()
        train_losses = []

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = classifier(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        # Validation
        classifier.eval()
        val_iou_scores = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                val_iou_scores.append(compute_iou(outputs, labels))

        mean_iou = np.mean(val_iou_scores)

        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {np.mean(train_losses):.4f}")
        print(f"Val IoU:    {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(classifier.state_dict(), "best_model.pth")
            print("Best model updated!")

        scheduler.step()

    torch.save(classifier.state_dict(), "last_model.pth")
    print("\nTraining Complete!")
    print(f"Best IoU Achieved: {best_iou:.4f}")


if __name__ == "__main__":
    main()