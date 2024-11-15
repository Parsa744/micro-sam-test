import os
import torch
from torch_em.model import UNETR
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
# Parameters (Ensure these match the training configuration)
img_size = 512  # Adjust based on your input image size during training
backbone = 'mae'  # Must match the training configuration
encoder_type = 'vit_b'  # Options: 'vit_b', 'vit_l', 'vit_h'
out_channels = 1  # Set to the number of classes or 1 for binary segmentation
use_skip_connection = True
use_conv_transpose = True  # Use ConvTranspose2d in upsampling
checkpoint_path = 'unetr_epoch_20.pth'

# Initialize the model (matching training parameters)
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
model.to(device)

# Load the trained weights
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()  # Set model to evaluation mode

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Function for inference on a single image
def infer_single_image(image):
    # Preprocess the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move image to the device
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Apply a sigmoid function to get the probability map (for binary segmentation)
    output = torch.sigmoid(output)

    # Convert to CPU and remove batch dimension
    output = output.squeeze().cpu().numpy()

    # Binarize the output mask (0.5 threshold)
    #output_mask = (output > 0.5).astype(np.uint8)

    return output

# Function to process a folder of images
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_mask.png")

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Perform inference
        predicted_mask = infer_single_image(image)

        # Save the predicted mask
        mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))  # Convert binary mask to 0-255
        mask_image.save(output_path)

        # Optionally, visualize each image and mask
        #visualize_inference(image, predicted_mask)

# Function to visualize the original image and its corresponding predicted mask
def visualize_inference(original_image, output_mask):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

# Example usage
input_folder = 'test/imgs'  # Replace with the path to your folder of test images
output_folder = 'test/pred_unetr_512_20'  # Replace with the path where you want to save output masks


start_time = time.time()
process_folder(input_folder, output_folder)
end_time = time.time()
print(end_time - start_time)
print(end_time - start_time / 1662)