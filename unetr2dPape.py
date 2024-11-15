# Train a 2D UNETR model for foreground and boundary segmentation on GPU
import os
from tokenize import endpats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_em.model import UNETR
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Parameters
img_size = 512  # Adjust based on your input image size
backbone = 'mae'  # Choose 'sam' or 'mae' depending on your use case
encoder_type = 'vit_b'  # Options: 'vit_b', 'vit_l', 'vit_h'
out_channels = 1  # Set to the number of classes or 1 for binary segmentation
use_skip_connection = True
use_conv_transpose = True  # Use ConvTranspose2d in upsampling
num_epochs = 20
batch_size = 4  # Adjust based on your GPU memory
learning_rate = 1e-4

# Initialize the model
model = UNETR(
    img_size=img_size,
    backbone=backbone,
    encoder=encoder_type,
    out_channels=out_channels,
    use_skip_connection=use_skip_connection,
    use_conv_transpose=use_conv_transpose,
)

# Check for GPU availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())
model.to(device)
# end program

# Custom Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.transform = transform
        self.target_transform = target_transform

        # Supported mask file extensions
        mask_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']

        # Create a mapping from image filenames to mask filenames
        self.image_to_mask = {}
        for img_file in self.image_files:
            img_name = os.path.splitext(img_file)[0]  # Get the base filename without extension

            # Search for the corresponding mask file with any of the supported extensions
            mask_file = None
            for ext in mask_extensions:
                potential_mask_name = f"{img_name}_masks{ext}"
                potential_mask_path = os.path.join(self.masks_dir, potential_mask_name)
                if os.path.exists(potential_mask_path):
                    mask_file = potential_mask_name
                    break

            if mask_file is not None:
                self.image_to_mask[img_file] = mask_file
            else:
                raise FileNotFoundError(f"No mask file found for image {img_file} with supported extensions {mask_extensions}")

        # Update the list of image files to only include those that have corresponding masks
        self.image_files = list(self.image_to_mask.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.image_to_mask[img_file]

        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Assuming masks are grayscale

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

# Transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# For masks, ensure they are tensors and scaled appropriately
target_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Directories containing images and masks
images_dir = 'data/imgs'
masks_dir = 'data/masks'

# Create dataset and dataloader
dataset = SegmentationDataset(images_dir, masks_dir, transform=transform, target_transform=target_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        # Move data to GPU
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        # Resize masks to match output size if necessary
        if outputs.size() != masks.size():
            masks = nn.functional.interpolate(masks, size=outputs.shape[2:], mode='nearest')

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Optionally, save the model checkpoint every few epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'unetr_epoch_{epoch+1}.pth')

print('Training complete.')
