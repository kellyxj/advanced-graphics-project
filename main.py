from typing import Optional
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from all_stuff import *

"""
Code from: https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX?ref=morioh.com&utm_source=morioh.com#scrollTo=EHNwlsOT7NTp
"""

# Seed RNG, for repeatability
seed = 9458
torch.manual_seed(seed)
np.random.seed(seed)

# Download sample data used in the official tiny_nerf example
if not os.path.exists('data/tiny_nerf_data.npz'):
    print("Downloading dataset...")
    os.system("wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz")
    print("Done.")

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU:", torch.cuda.is_available())

# Load input images, poses, and intrinsics
data = np.load("tiny_nerf_data.npz")

# Images
images = data["images"]

# Camera extrinsics (poses)
tform_cam2world = data["poses"]
tform_cam2world = torch.from_numpy(tform_cam2world).to(device)

# Focal length (intrinsics)
focal_length = data["focal"]
focal_length = torch.from_numpy(focal_length).to(device)

# Height and width of each image
height, width = images.shape[1:3]

# Near and far clipping thresholds for depth values.
near_thresh = 2.
far_thresh = 6.

# Hold one image out (for test).
testimg, testpose = images[101], tform_cam2world[101]
testimg = torch.from_numpy(testimg).to(device)

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

# Initialize model
print("Initialize model")

# Define model and optimizer
model = VeryTinyNerfModel()
model.to(device)


# Train
print("Training")

train_tinynerf(
    model,
    images,
    tform_cam2world,
    4,
    height, width, focal_length, near_thresh, far_thresh,
    device
)