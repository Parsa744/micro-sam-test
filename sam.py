#mamba activate micro-sam-new
import os
from glob import glob
import h5py
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components
from torch_em.util.util import get_random_colors
import torch_em.data.datasets.light_microscopy.covid_if as covid_if
from micro_sam import util
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import time
# Define the function to assign a unique color to each shade in an image
def assign_unique_colors_to_shades(image_np):
    """
    Assign a unique color to each shade in a grayscale image.

    Args:
        image_np (numpy.ndarray): Grayscale image as a 2D numpy array.

    Returns:
        color_image (numpy.ndarray): RGB image with unique color for each shade.
    """
    # Ensure the image is in grayscale format
    if len(image_np.shape) == 3:
        image_np = image_np[:, :, 0]  # If HWC, take a single channel

    # Get unique shades in the image
    unique_shades = np.unique(image_np)

    # Create a color map where each unique grayscale value is assigned a random RGB color
    color_map = {shade: [random.randint(0, 255) for _ in range(3)] for shade in unique_shades}

    # Create an RGB image to assign the unique color to each shade
    color_image = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)

    # Map each pixel in the original grayscale image to the new color
    for shade in unique_shades:
        color_image[image_np == shade] = color_map[shade]

    return color_image


def run_automatic_instance_segmentation(image, model_type="vit_b_lm"):
    """Automatic Instance Segmentation by training an additional instance decoder in SAM.



    NOTE: It is supported only for `µsam` models.



    Args:

        image: The input image.

        model_type: The choice of the `µsam` model.



    Returns:

        The instance segmentation.

    """

    # Step 1: Initialize the model attributes using the pretrained µsam model weights.

    #   - the 'predictor' object for generating predictions using the Segment Anything model.

    #   - the 'decoder' backbone (for AIS).

    predictor, decoder = get_predictor_and_decoder(

        model_type=model_type,  # choice of the Segment Anything model

        checkpoint_path=None,  # overwrite to pass our own finetuned model

    )

    # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.

    image_embeddings = util.precompute_image_embeddings(

        predictor=predictor,  # the predictor object responsible for generating predictions

        input_=image,  # the input image

        ndim=2,  # number of input dimensions

    )

    # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.

    ais = InstanceSegmentationWithDecoder(predictor, decoder)

    # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.

    ais.initialize(

        image=image,  # the input image

        image_embeddings=image_embeddings,  # precomputed image embeddings

    )

    # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.

    prediction = ais.generate()

    prediction = mask_data_to_segmentation(prediction, with_background=True)

    return prediction


def run_automatic_mask_generation(image, model_type="vit_b"):

    predictor = util.get_sam_model(
        model_type=model_type,  # choice of the Segment Anything model
    )
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,  # the predictor object responsible for generating predictions
        input_=image,  # the input image
        ndim=2,  # number of input dimensions

    )

    amg = AutomaticMaskGenerator(predictor)

    amg.initialize(
        image=image,  # the input image
        image_embeddings=image_embeddings,  # precomputed image embeddings
    )

    prediction = amg.generate(
        pred_iou_thresh=0.75,
        stability_score_thresh=0.75)
    min_area_threshold = 5
    prediction = mask_data_to_segmentation(prediction, with_background=True)

    filtered_prediction = filter_small_masks(prediction, min_area_threshold)

    return filtered_prediction


def filter_small_masks(prediction, min_area_threshold):
    unique_masks = np.unique(prediction)
    filtered_prediction = np.zeros_like(prediction)

    for mask_id in unique_masks:
        if mask_id == 0:  # Skip background
            continue

        mask_area = np.sum(prediction == mask_id)

        if mask_area >= min_area_threshold:
            filtered_prediction[prediction == mask_id] = mask_id

    return filtered_prediction




class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_paths = sorted(glob(os.path.join(images_dir, "*.png")))  # Load images

        # Load masks with either .tif or .png extensions
        mask_paths_tif = {os.path.basename(p).replace(".png", "_masks.tif"): p for p in
                          glob(os.path.join(masks_dir, "*.tif"))}
        mask_paths_png = {os.path.basename(p).replace(".png", "_masks.png"): p for p in
                          glob(os.path.join(masks_dir, "*.png"))}

        # Combine both dictionaries into one for masks
        self.mask_paths = {**mask_paths_tif, **mask_paths_png}

        # Ensure every image has a corresponding mask
        self.paired_data = [(img_path, self.mask_paths[os.path.basename(img_path).replace(".png", "_masks.tif")])
                            if os.path.basename(img_path).replace(".png", "_masks.tif") in self.mask_paths
                            else (img_path, self.mask_paths[os.path.basename(img_path).replace(".png", "_masks.png")])
                            for img_path in self.image_paths if (
                                    os.path.basename(img_path).replace(".png", "_masks.tif") in self.mask_paths or
                                    os.path.basename(img_path).replace(".png", "_masks.png") in self.mask_paths
                            )]

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        img_path, mask_path = self.paired_data[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming the mask is single-channel

        # Apply transformations, if any
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Specify transformations if needed, e.g., resizing, normalization, etc.
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
# Initialize the dataset and dataloader
images_dir = 'test/imgs'
masks_dir = 'test/masks'
dataset = SegmentationDataset(images_dir, masks_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model_choice = "vit_b_lm"
output_folder = "vit_b_lm_colored"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Wrap the dataloader with tqdm to display progress
for idx, (image, gt) in enumerate(tqdm(dataloader, desc="Processing Images")):
    # Get the original image filename
    original_image_path = dataset.image_paths[idx]
    original_filename = os.path.basename(original_image_path)
    filename, ext = os.path.splitext(original_filename)

    # Convert the image tensor to the expected format for the model (e.g., numpy array)
    image_np = image[0].permute(1, 2, 0).numpy()  # Convert first image in batch to HWC format

    # Run segmentation prediction
    start = time.time()
    prediction = run_automatic_mask_generation(image_np, model_type=model_choice)
    end = time.time()
    print(f'Automatic mask generation took {end - start:.2f} seconds')
    # Display image, ground truth, and prediction
    colored = assign_unique_colors_to_shades(prediction)

    # Save the colored prediction in the output folder with "_pred" appended
    output_path = os.path.join(output_folder, f"{filename}_pred.png")
    colored_image = Image.fromarray(np.uint8(colored))  # Assuming the output is in a compatible format
    colored_image.save(output_path)
