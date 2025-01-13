import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from datasets import load_dataset
from PIL import Image, ImageOps

def extract_points_from_mask(mask, num_points):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    mask_indices = np.argwhere(mask > 0)
    if len(mask_indices) <= num_points:
        selected_indices = mask_indices
    else:
        selected_indices = mask_indices[np.random.choice(len(mask_indices), num_points, replace=False)]
    height, width = mask.shape
    normalized_points = selected_indices[:, ::-1] / [width, height]
    return normalized_points


def visualize_points_on_mask(mask):
    points = extract_points_from_mask(mask, 3)
    mask = np.array(mask)
    height, width = mask.shape
    pixel_points = (points * [width, height]).astype(int)
    plt.imshow(mask)
    plt.scatter(pixel_points[:, 0], pixel_points[:, 1], color="red", marker="o")
    plt.axis("off")
    plt.savefig("points.png")

raw_dataset = load_dataset("kowndinya23/Kvasir-SEG")
train_set = raw_dataset["train"]
mask_gt = train_set[0]["annotation"]


visualize_points_on_mask(mask_gt)



def generate_masks_with_points(num_points=3):
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    raw_dataset = load_dataset("kowndinya23/Kvasir-SEG")
    train_set = raw_dataset["train"]

    sample_idxs = [0]


    
    num_predicted_masks = 3
    fig, axs = plt.subplots(len(sample_idxs), 2 + num_predicted_masks, figsize=(15, 5 * len(sample_idxs)))

    for row, idx in enumerate(sample_idxs):
        image = train_set[idx]["image"]
        image = np.array(image.convert("RGB"))

        mask_gt = train_set[idx]["annotation"]

        points = extract_points_from_mask(mask_gt, num_points)
        
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=None,
            point_grids=[points]
        )

    
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_masks = mask_generator.generate(image)


        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask_gt)
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        for col, mask in enumerate(predicted_masks[:num_predicted_masks]):
            axs[2 + col].imshow(mask)
            axs[2 + col].set_title(f"Predicted Mask {col + 1}")
            axs[2 + col].axis("off")

    plt.savefig("batu-simge/predicted_masks.png")
    plt.show()


# generate_masks_with_points(num_points=3)

