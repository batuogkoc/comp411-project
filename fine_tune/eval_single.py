import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from PIL import Image, ImageOps


# Paths
image_id = "cju0qkwl35piu0993l0dewei2"

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
image_path = f"fine_tune/Kvasir-SEG/images/{image_id}.jpg"
mask_path = f"fine_tune/Kvasir-SEG/masks/{image_id}.jpg"

# Load model and predictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load image
image = Image.open(image_path)
width, height = image.size

# Predict mask
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(box=np.array([0, 0, width, height]))

predicted_mask = (masks[0] > 0.5).astype(np.uint8)  # Binary mask

# Load ground truth mask
ground_truth_mask = np.array(Image.open(mask_path).convert("L"))
ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)  # Binary mask

# Ensure same shape
if predicted_mask.shape != ground_truth_mask.shape:
    raise ValueError("Predicted and ground truth masks have different shapes.")

# Calculate IoU
intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
union = np.logical_or(predicted_mask, ground_truth_mask).sum()
iou = intersection / union

print(f"IoU for image {image_path}: {iou:.4f}")



# Convert the predicted mask to an image
predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))

# Convert the ground truth mask to an image
ground_truth_mask_image = Image.fromarray((ground_truth_mask * 255).astype(np.uint8))

# Add borders to each image for better separation (optional)
predicted_mask_image = ImageOps.expand(predicted_mask_image, border=5, fill='black')
ground_truth_mask_image = ImageOps.expand(ground_truth_mask_image, border=5, fill='black')

# Combine the two images side by side
combined_width = predicted_mask_image.width + ground_truth_mask_image.width
combined_height = max(predicted_mask_image.height, ground_truth_mask_image.height)

combined_image = Image.new("RGB", (combined_width, combined_height))

# Paste the images into the combined canvas
combined_image.paste(predicted_mask_image.convert("RGB"), (0, 0))
combined_image.paste(ground_truth_mask_image.convert("RGB"), (predicted_mask_image.width, 0))

# Save the combined image
combined_image.save("fine_tune/combined_masks.jpg", "JPEG")


