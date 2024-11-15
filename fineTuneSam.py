import warnings
warnings.filterwarnings("ignore")
from glob import glob
from IPython.display import FileLink
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components
import torch
from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors
import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from PIL import Image
import glob
import os

# Update the paths for your image and segmentation mask directories
image_dir = "test/imgs"
segmentation_dir = "test/masks"

# Create a list of all supported image formats (png, tif, jpg)
image_paths = glob.glob(os.path.join(image_dir, "*.tif")) + \
              glob.glob(os.path.join(image_dir, "*.png")) + \
              glob.glob(os.path.join(image_dir, "*.jpg"))

mask_paths = glob.glob(os.path.join(segmentation_dir, "*_masks.tif")) + \
             glob.glob(os.path.join(segmentation_dir, "*_masks.png")) + \
             glob.glob(os.path.join(segmentation_dir, "*_masks.jpg"))

# Convert all RGB images to grayscale to ensure correct input shape
for image_path in image_paths:
    img = Image.open(image_path)
    if img.mode != 'L':  # Convert to grayscale if not already
        img = img.convert('L')
        img.save(image_path)

for mask_path in mask_paths:
    mask = Image.open(mask_path)
    if mask.mode != 'L':
        mask = mask.convert('L')
        mask.save(mask_path)

# Training parameters
batch_size = 1  # Adjust the batch size if needed
patch_shape = ( 512, 512)  # Define the patch shape, assuming 2D images

# Enable training for instance segmentation
train_instance_segmentation = True

# Define the sampler to ensure at least one foreground instance per input
sampler = MinInstanceSampler(min_size=25)

# Creating the data loader for training
train_loader = sam_training.default_sam_loader(
    raw_paths=image_paths,  # Explicitly pass the list of paths
    raw_key=None,  # Set to None because paths are explicitly provided
    label_paths=mask_paths,  # Explicitly pass the list of mask paths
    label_key=None,  # Set to None because paths are explicitly provided
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

# Creating the data loader for validation (using same approach as training)
val_loader = sam_training.default_sam_loader(
    raw_paths=image_paths,  # Explicitly pass the list of paths
    raw_key=None,  # Set to None because paths are explicitly provided
    label_paths=mask_paths,  # Explicitly pass the list of mask paths
    label_key=None,  # Set to None because paths are explicitly provided
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

#check_loader(train_loader, 4, plt=True) # it is working properly !
n_objects_per_batch = 5  # the number of objects per batch that will be sampled
device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
print(device)
n_epochs = 5  # how long we train (in epochs)
model_type = "vit_b"
checkpoint_name = "sam_hela"
root_dir = ''

sam_training.train_sam(
    name=checkpoint_name,
    save_root=os.path.join(root_dir, "models"),
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    n_objects_per_batch=n_objects_per_batch,
    with_segmentation_decoder=train_instance_segmentation,
    device=device,
)