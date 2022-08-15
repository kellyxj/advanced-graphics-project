from typing import Optional
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import *
from src.training import *
from src.utils import *

"""
Code from: https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX?ref=morioh.com&utm_source=morioh.com#scrollTo=EHNwlsOT7NTp
"""

# Seed RNG, for repeatability
seed = 9458
torch.manual_seed(seed)
np.random.seed(seed)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU:", torch.cuda.is_available())

# Download sample data used in the official tiny_nerf example
if not os.path.exists('data/tiny_nerf_data.npz'):
    print("Downloading dataset...")
    os.system("wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz -P data")
    print("Done.")

images, tform_cam2world, height, width, focal_length, near_thresh, far_thresh = load_data("data/tiny_nerf_data.npz", device)

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