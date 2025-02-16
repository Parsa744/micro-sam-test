import os
import random
import time
import numpy as np
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from celery import shared_task
from celery.exceptions import Retry

# micro-sam imports
from micro_sam import util
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def assign_unique_colors_to_shades(image_np):
    """
    Assign a unique color to each shade in a grayscale image.

    Args:
        image_np (numpy.ndarray): Grayscale image as a 2D numpy array.

    Returns:
        color_image (numpy.ndarray): RGB image with unique color for each shade.
    """
    # Ensure the image is grayscale if it has multiple channels
    if len(image_np.shape) == 3:
        image_np = image_np[:, :, 0]

    # Get unique shades in the image
    unique_shades = np.unique(image_np)

    # Create a color map (random RGB) for each unique grayscale value
    color_map = {shade: [random.randint(0, 255) for _ in range(3)] for shade in unique_shades}

    # Create an RGB image with the new colors
    color_image = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
    for shade in unique_shades:
        color_image[image_np == shade] = color_map[shade]

    return color_image


def run_automatic_instance_segmentation(image, model_type="vit_b_lm"):
    """
    Automatic Instance Segmentation by training an additional instance decoder in SAM.
    NOTE: It is supported only for µsam models.

    Args:
        image (numpy.ndarray): The input image in HWC format (RGB).
        model_type (str): The choice of the µsam model.

    Returns:
        numpy.ndarray: Instance segmentation (label image).
    """
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,
        checkpoint_path=None
    )

    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,
        input_=image,
        ndim=2
    )

    ais = InstanceSegmentationWithDecoder(predictor, decoder)
    ais.initialize(
        image=image,
        image_embeddings=image_embeddings
    )
    prediction = ais.generate()
    prediction = mask_data_to_segmentation(prediction, with_background=True)
    return prediction


def run_automatic_mask_generation(image, model_type="vit_b"):
    """
    Automatic mask generation using SAM.

    Args:
        image (numpy.ndarray): The input image in HWC format (RGB).
        model_type (str): The choice of the SAM model.

    Returns:
        numpy.ndarray: Label image with automatically generated masks.
    """
    predictor = util.get_sam_model(model_type=model_type)
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,
        input_=image,
        ndim=2
    )

    amg = AutomaticMaskGenerator(predictor)
    amg.initialize(
        image=image,
        image_embeddings=image_embeddings
    )

    prediction = amg.generate(pred_iou_thresh=0.75, stability_score_thresh=0.75)
    prediction = mask_data_to_segmentation(prediction, with_background=True)
    filtered_prediction = filter_small_masks(prediction, min_area_threshold=5)
    return filtered_prediction


def filter_small_masks(prediction, min_area_threshold):
    """
    Filter out masks that have an area smaller than 'min_area_threshold'.

    Args:
        prediction (numpy.ndarray): Label image of the segmentation.
        min_area_threshold (int): Minimum area to keep a mask.

    Returns:
        numpy.ndarray: Filtered label image.
    """
    unique_masks = np.unique(prediction)
    filtered_prediction = np.zeros_like(prediction)
    for mask_id in unique_masks:
        if mask_id == 0:  # Skip background
            continue
        mask_area = np.sum(prediction == mask_id)
        if mask_area >= min_area_threshold:
            filtered_prediction[prediction == mask_id] = mask_id
    return filtered_prediction


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Dataset that can optionally load masks if masks_dir is provided.
    If masks_dir is None, only the images are loaded (masks are None).
    """
    def __init__(self, images_dir, masks_dir=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.has_masks = masks_dir is not None

        self.image_paths = sorted(glob(os.path.join(images_dir, "*.png")))

        if self.has_masks:
            # Load possible mask paths (.tif or .png)
            mask_paths_tif = {
                os.path.basename(p).replace(".png", "_masks.tif"): p
                for p in glob(os.path.join(masks_dir, "*.tif"))
            }
            mask_paths_png = {
                os.path.basename(p).replace(".png", "_masks.png"): p
                for p in glob(os.path.join(masks_dir, "*.png"))
            }
            self.mask_paths = {**mask_paths_tif, **mask_paths_png}

            self.paired_data = []
            for img_path in self.image_paths:
                basename = os.path.basename(img_path)
                tif_key = basename.replace(".png", "_masks.tif")
                png_key = basename.replace(".png", "_masks.png")
                if tif_key in self.mask_paths:
                    self.paired_data.append((img_path, self.mask_paths[tif_key]))
                elif png_key in self.mask_paths:
                    self.paired_data.append((img_path, self.mask_paths[png_key]))
                else:
                    # No matching mask found
                    self.paired_data.append((img_path, None))
        else:
            # If no masks_dir provided, store only images
            self.paired_data = [(img, None) for img in self.image_paths]

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        img_path, mask_path = self.paired_data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # If we have masks_dir and a valid mask path, load mask
        if self.has_masks and mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                mask = self.transform(mask)
        else:
            mask = None  # No mask available

        return image, mask


# --------------------------------------------------
# Custom Collate Function to Handle None Masks
# --------------------------------------------------

def custom_collate_fn(batch):
    """
    Collate function that handles (image, mask=None) by converting None masks
    to a dummy (zero) tensor, or optionally leaving them in a list.
    """
    images = []
    masks = []

    for image, mask in batch:
        images.append(image)
        # If the mask is None, let's make a zero-tensor with shape (1, H, W)
        # so that we don't break the default_collate
        if mask is None:
            # Create a dummy mask of shape (1, H, W) for a single image channel
            dummy_mask = torch.zeros((1, image.shape[1], image.shape[2]))
            masks.append(dummy_mask)
        else:
            masks.append(mask)

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks


# --------------------------------------------------
# Segmentation Pipeline
# --------------------------------------------------

def run_segmentation_inference(
    images_dir,
    masks_dir=None,
    output_folder="data/output",
    model_choice="vit_b_lm",
    use_instance_segmentation=True,
    batch_size=1,
    transform_size=(512, 512)
):
    """
    Runs the segmentation inference pipeline.

    Args:
        images_dir (str): Directory containing input images.
        masks_dir (str, optional): Directory containing ground truth masks. If None, masks are not loaded.
        output_folder (str): Directory to save segmentation results.
        model_choice (str): SAM model choice (e.g., 'vit_b_lm', 'vit_b').
        use_instance_segmentation (bool): Whether to use instance segmentation or mask generation.
        batch_size (int): Batch size for data loading.
        transform_size (tuple): Size (H, W) to resize images for inference.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize(transform_size),
        transforms.ToTensor()
    ])

    # Dataset & DataLoader
    dataset = SegmentationDataset(images_dir=images_dir, masks_dir=masks_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn  # Use our custom collate function
    )

    # Inference loop
    for batch_idx, (image_batch, mask_batch) in enumerate(tqdm(dataloader, desc="Processing Images")):
        # image_batch: (B, 3, H, W)
        # mask_batch: (B, 1, H, W) or dummy if None

        for i in range(len(image_batch)):
            dataset_idx = batch_idx * batch_size + i
            if dataset_idx >= len(dataset.image_paths):
                break

            # Prepare for inference
            original_image_path = dataset.image_paths[dataset_idx]
            original_filename = os.path.basename(original_image_path)
            filename, _ = os.path.splitext(original_filename)
            image_np = image_batch[i].permute(1, 2, 0).numpy()  # HWC format

            start_time = time.time()
            if use_instance_segmentation:
                prediction = run_automatic_instance_segmentation(image_np, model_type=model_choice)
            else:
                prediction = run_automatic_mask_generation(image_np, model_type=model_choice)
            end_time = time.time()

            print(f"Segmentation took {end_time - start_time:.2f}s for {filename}")

            # Assign random colors to each label
            colored = assign_unique_colors_to_shades(prediction)

            # Save results
            output_path = os.path.join(output_folder, f"{filename}_pred.png")
            colored_image = Image.fromarray(np.uint8(colored))
            colored_image.save(output_path)

def filter_small_masks(prediction, min_area_threshold):
    """
    Filter out masks that have an area smaller than 'min_area_threshold'.

    Args:
        prediction (numpy.ndarray): Label image of the segmentation.
        min_area_threshold (int): Minimum area to keep a mask.

    Returns:
        numpy.ndarray: Filtered label image.
    """
    unique_labels = np.unique(prediction)
    filtered = np.zeros_like(prediction)
    for mask_id in unique_labels:
        if mask_id == 0:  # Skip background
            continue
        if np.sum(prediction == mask_id) >= min_area_threshold:
            filtered[prediction == mask_id] = mask_id
    return filtered

@shared_task(max_retries=3, default_retry_delay=10)
def run_segmentation_inference_for_one_image(
    image_dir,
    mask_dir=None,            # Optional: Not used in the code below, but provided for flexibility
    output=None,
    model_choice="vit_b_lm",
    use_instance_segmentation=True,
    transform_size=(512, 512)
):
    """
    Perform segmentation inference for a single image.

    Args:
        image_dir (str): Path to the input image.
        mask_dir (str, optional): Path to the corresponding mask, if available (not used here).
        output (str): Path to save the output segmentation result.
        model_choice (str): Choice of the SAM model (e.g., 'vit_b_lm', 'vit_b').
        use_instance_segmentation (bool): Whether to use instance segmentation or mask generation.
        transform_size (tuple): (height, width) to resize the image before inference.
    """
    # Device check (not strictly necessary for single image, but good practice)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if output is None:
            # output is same as input with '_pred' appended
            output = 'output_pred.png'


        # Step 1: Load and transform the image
        transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor()
        ])
        image_pil = Image.open(image_dir).convert("RGB")
        image_tensor = transform(image_pil).to(device)  # (3, H, W)

        # Convert to numpy for the micro-sam functions (HWC)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Step 2: Run Segmentation
        start_time = time.time()
        if use_instance_segmentation:
            prediction = run_automatic_instance_segmentation(image_np, model_type=model_choice)
        else:
            prediction = run_automatic_mask_generation(image_np, model_type=model_choice)
        end_time = time.time()


        # Step 3: Color the segmentation
        colored = assign_unique_colors_to_shades(prediction)
        colored_image = Image.fromarray(np.uint8(colored))

        # Step 4: Save the result
        colored_image.save(output)
        return output
    except Exception as e:
        return Retry(exc=e)
# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    result = run_segmentation_inference_for_one_image(
        image_dir="data/imgs/2019_par_tmc11_tmc11_000.png",
        mask_dir=None,  # or "path/to/your_mask.png" if you have it, though not used in this function
        output="data/output/2019_par_tmc11_tmc11_000_pred.png",
        model_choice="vit_b_lm",
        use_instance_segmentation=True,
        transform_size=(512, 512)
    )
    return result
    '''
    images_dir = "data/imgs"
    # Set masks_dir=None if you do not have masks
    masks_dir = None
    output_folder = "data/output"
    model_choice = "vit_b_lm"
    use_instance_segmentation = True
    batch_size = 1
    transform_size = (512, 512)

    run_segmentation_inference(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_folder=output_folder,
        model_choice=model_choice,
        use_instance_segmentation=use_instance_segmentation,
        batch_size=batch_size,
        transform_size=transform_size
    )'''


if __name__ == "__main__":
    main()
