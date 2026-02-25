import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

from improved_training import SegmentationHeadConvNeXt, n_classes

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "best_model.pth"
IMAGE_PATH =  "C:/Users/lucky/OneDrive/Desktop/dualityai/Offroad_Segmentation_testImages/Color_Images/0000152.png"#change to your test image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load Backbone
# ------------------------------
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(device)

# ------------------------------
# Prepare Image
# ------------------------------
w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ------------------------------
# Create Model
# ------------------------------
with torch.no_grad():
    features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]

n_embedding = features.shape[2]

model = SegmentationHeadConvNeXt(
    in_channels=n_embedding,
    out_channels=n_classes,
    tokenW=w // 14,
    tokenH=h // 14
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ------------------------------
# Predict
# ------------------------------
with torch.no_grad():
    logits = model(features)
    outputs = F.interpolate(logits, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    prediction = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

print("Prediction shape:", prediction.shape)
np.save("prediction_mask.npy", prediction)
print("Prediction saved as prediction_mask.npy")