import os
import zipfile
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from tqdm import tqdm

# Paths
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
zip_path = "fine_tune/kvasir-seg.zip"
extract_path = "fine_tune"
images_path = os.path.join(extract_path, "Kvasir-SEG/images")
masks_path = os.path.join(extract_path, "Kvasir-SEG/masks")

# Extract the dataset if not already extracted
if not os.path.exists(images_path) or not os.path.exists(masks_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

# Load SAM2 model
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Initialize variables for IoU calculation
ious = []

# Process dataset
image_files = os.listdir(images_path)
mask_files = os.listdir(masks_path)

# Sort files to ensure consistent pairing
image_files.sort()
mask_files.sort()

print("Processing dataset...")
for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
    # Load the image and ground truth mask
    image_path = os.path.join(images_path, image_file)
    mask_path = os.path.join(masks_path, mask_file)

    image = Image.open(image_path).convert("RGB")
    ground_truth_mask = np.array(Image.open(mask_path).convert("L"))

    # Ensure ground truth mask is binary
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Run SAM2 predictor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        width, height = image.size
        masks, _, _ = predictor.predict(box=np.array([0, 0, width, height]))

    # Convert predicted mask to binary
    predicted_mask = (masks[0] > 0.5).astype(np.uint8)

    # Resize predicted mask to match the shape of the ground truth mask
    predicted_mask_resized = np.array(
        Image.fromarray(predicted_mask).resize(ground_truth_mask.shape[::-1], resample=Image.NEAREST)
    )

    # Calculate IoU using NumPy
    intersection = np.logical_and(predicted_mask_resized, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask_resized, ground_truth_mask).sum()
    iou = intersection / union if union > 0 else 0
    ious.append(iou)

# Calculate average IoU
average_iou = np.mean(ious)
print(f"Average IoU: {average_iou:.4f}")
