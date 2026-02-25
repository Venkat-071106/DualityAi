import numpy as np
import matplotlib.pyplot as plt

mask = np.load("prediction_mask.npy")

# 10 class color map
colors = np.array([
    [0, 0, 0],        # background
    [0, 128, 0],      # trees
    [34, 139, 34],    # lush bushes
    [189, 183, 107],  # dry grass
    [139, 69, 19],    # dry bushes
    [160, 82, 45],    # ground clutter
    [105, 105, 105],  # logs
    [112, 128, 144],  # rocks
    [70, 130, 180],   # landscape
    [135, 206, 235],  # sky
])

colored_mask = colors[mask]

plt.imshow(colored_mask)
plt.title("Predicted Segmentation")
plt.axis("off")
plt.show()