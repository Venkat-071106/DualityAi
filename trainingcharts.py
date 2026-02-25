import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Example: manually input your values if not saved
train_loss = [0.45, 0.43, 0.41, 0.40, 0.39]
val_iou = [0.48, 0.49, 0.47, 0.47, 0.50]

os.makedirs("results", exist_ok=True)

# Loss Curve
plt.figure()
plt.plot(train_loss)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("results/loss_curve.png")
plt.close()

# IoU Curve
plt.figure()
plt.plot(val_iou)
plt.title("Validation IoU Curve")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.grid(True)
plt.savefig("results/iou_curve.png")
plt.close()

print("Charts saved inside results/")