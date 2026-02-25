import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm

from improved_training import MaskDataset, SegmentationHeadConvNeXt, compute_iou, n_classes


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset paths
    data_dir = "C:/Users/lucky/OneDrive/Desktop/dualityai/Offroad_Segmentation_Training_Dataset/train"
    val_dir  = "C:/Users/lucky/OneDrive/Desktop/dualityai/Offroad_Segmentation_Training_Dataset/val"

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

    trainset = MaskDataset(data_dir, transform, mask_transform)
    valset   = MaskDataset(val_dir, transform, mask_transform)

    train_loader = DataLoader(trainset, batch_size=2, shuffle=True)
    val_loader   = DataLoader(valset, batch_size=2, shuffle=False)

    # Load backbone
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

    # 🔥 Load previous BEST model
    classifier.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Loaded best_model.pth for fine-tuning")

    # 🔥 Lower learning rate
    lr = 2e-5
    n_epochs = 20

    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0

    print("\nStarting Stage 2 Fine-Tuning...\n")

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

        #Overwrite same best_model.pth if improved
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(classifier.state_dict(), "best_model.pth")
            print("best_model.pth updated!")

        scheduler.step()

    print("\nStage 2 Fine-Tuning Complete!")
    print("Best IoU in Stage 2:", best_iou)


if __name__ == "__main__":
    main()