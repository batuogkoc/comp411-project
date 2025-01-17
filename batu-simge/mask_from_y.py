import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from datasets import load_dataset
from PIL import Image, ImageOps
from torch_datasets import KvasirSEGDataset
import albumentations as A
from torch.utils.data import DataLoader


def extract_points_from_mask(mask, num_points, normalize=False):
    if not isinstance(mask, np.ndarray):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        else:
            mask = np.array(mask)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    mask_indices = np.argwhere(mask > 0)
    if len(mask_indices) <= num_points:
        selected_indices = mask_indices
    else:
        selected_indices = mask_indices[
            np.random.choice(len(mask_indices), num_points, replace=False)
        ]
    height, width = mask.shape
    if normalize:
        return np.flip(selected_indices, -1) / [width, height]
    else:
        return np.flip(selected_indices, -1).copy()


def extract_points_from_mask_batched(masks, num_points, normalize=False):
    points = []
    for mask in masks:
        points.append(extract_points_from_mask(mask, num_points, normalize=normalize))

    return points


def visualize_points_on_mask(mask):
    print(np.unique(mask))
    points = extract_points_from_mask(mask, 3)
    mask = np.array(mask)
    height, width = mask.shape
    pixel_points = (points * [width, height]).astype(int)
    plt.imshow(mask)
    plt.scatter(pixel_points[:, 0], pixel_points[:, 1], color="red", marker="o")
    plt.axis("off")
    plt.savefig("points.png")


def generate_masks_with_points(predictor, dataset, num_points=3, num_samples=5):
    num_predicted_masks = 3
    fig, axs = plt.subplots(
        num_samples, 2 + num_predicted_masks, figsize=(15, 5 * num_samples)
    )

    for row, (image, mask_gt) in enumerate([dataset[i] for i in range(num_samples)]):
        mask_gt = (mask_gt > 0).astype(np.float32)

        points = extract_points_from_mask(mask_gt, num_points)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            predicted_masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
            )
            predicted_masks = predicted_masks.detach().cpu().numpy()
        axs[row, 0].imshow(image)
        axs[row, 0].set_title("Original Image")
        axs[row, 0].scatter(
            points[:, 0],
            points[:, 1],
            color="red",
            marker="o",
        )
        axs[row, 0].axis("off")

        axs[row, 1].imshow(mask_gt)
        axs[row, 1].set_title("Ground Truth Mask")
        axs[row, 1].axis("off")

        for col, mask in enumerate(predicted_masks[:num_predicted_masks]):
            axs[row, 2 + col].imshow(mask)
            axs[row, 2 + col].set_title(f"Predicted Mask {col + 1}")
            axs[row, 2 + col].axis("off")

    return fig


def _visualize_test():
    mask_gt = KvasirSEGDataset("train")[0][1]
    visualize_points_on_mask(mask_gt)


def _multi_visualize_test():
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    dataset = KvasirSEGDataset("train", transform=A.Compose([A.Resize(1024, 1024)]))
    fig = generate_masks_with_points(predictor, dataset, 10)
    fig.savefig("batu-simge/predicted_masks.png")


# generate_masks_with_points(num_points=3)
if __name__ == "__main__":
    # _visualize_test()
    _multi_visualize_test()
