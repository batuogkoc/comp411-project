import os
import zipfile
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score

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

    width, height = image.size

    # Run SAM2 predictor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=np.array([0, 0, width, height]))

    # Convert predicted mask to binary
    predicted_mask = (masks[0] > 0.5).astype(np.uint8)  # Thresholding to binary

    # Convert ground truth mask to binary (ensure consistency)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    # Ensure both masks are flattened and of the same size
    predicted_mask_flat = predicted_mask.flatten()
    ground_truth_mask_flat = ground_truth_mask.flatten()

    # Ensure consistent sizes
    if predicted_mask_flat.size != ground_truth_mask_flat.size:
        print(f"Size mismatch detected: Predicted ({predicted_mask_flat.size}), Ground Truth ({ground_truth_mask_flat.size}). Resizing...")
        predicted_mask_flat = predicted_mask_flat[:ground_truth_mask_flat.size]
        ground_truth_mask_flat = ground_truth_mask_flat[:predicted_mask_flat.size]
   
   # Calculate IoU
    iou = jaccard_score(ground_truth_mask_flat, predicted_mask_flat, average="binary")
    ious.append(iou)

# Calculate average IoU
average_iou = np.mean(ious)
print(f"Average IoU: {average_iou:.4f}")
