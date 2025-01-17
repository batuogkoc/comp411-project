import torch
from torch import nn
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import numpy as np
from torch_datasets import KvasirSEGDataset
import albumentations as A
from matplotlib import pyplot as plt
from utils import extract_points_from_mask


class SAM2Modified(nn.Module):
    def __init__(self, predictor: SAM2ImagePredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image, point_coords=None, point_labels=None):
        normalize_coords = True
        multimask_output = True
        return_logits = True
        img_idx = -1
        bs = 1

        # set image
        self.predictor.set_image(image)
        # Transform input prompts

        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            point_coords,
            point_labels,
            box=None,
            mask_logits=None,
            normalize_coords=normalize_coords,
        )
        concat_points = None
        # _predict
        if point_coords is not None and point_labels is not None:
            concat_points = (point_coords, point_labels)

            sparse_embeddings, dense_embeddings = (
                self.predictor.model.sam_prompt_encoder(
                    points=concat_points,
                    boxes=None,
                    masks=mask_input,
                )
            )
        else:
            sparse_embeddings = torch.empty(
                (bs, 0, self.predictor.model.sam_prompt_embed_dim),
                device=self.predictor.model.device,
            )
            print(self.predictor.model.sam_prompt_encoder.image_embedding_size)
            dense_embeddings = (
                self.predictor.model.sam_prompt_encoder.no_mask_embed.weight.reshape(
                    1, -1, 1, 1
                ).expand(
                    (
                        bs,
                        -1,
                        self.predictor.model.sam_prompt_encoder.image_embedding_size[0],
                        self.predictor.model.sam_prompt_encoder.image_embedding_size[1],
                    )
                )
            )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self.predictor._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"][img_idx].unsqueeze(
                0
            ),
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self.predictor._transforms.postprocess_masks(
            low_res_masks, self.predictor._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.predictor.mask_threshold

        masks_np = masks.squeeze(0).float()
        iou_predictions_np = iou_predictions.squeeze(0).float()
        low_res_masks_np = low_res_masks.squeeze(0).float()
        # return masks_np, iou_predictions_np, low_res_masks_np
        return masks_np


def __generate_masks_with_points__(
    model: SAM2Modified, dataset, num_points=3, num_samples=5
):
    num_predicted_masks = 3
    fig, axs = plt.subplots(
        num_samples, 2 + num_predicted_masks, figsize=(15, 5 * num_samples)
    )

    for row, (image, mask_gt) in enumerate([dataset[i] for i in range(num_samples)]):
        mask_gt = (mask_gt > 0).astype(np.float32)

        points = extract_points_from_mask(mask_gt, num_points)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_masks = model(image)
            predicted_masks = predicted_masks.detach().cpu().numpy() > 0
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


if __name__ == "__main__":
    dataset = KvasirSEGDataset(
        "train",
        transform=A.Compose(
            [
                A.Resize(
                    height=1024,
                    width=1024,
                ),
            ]
        ),
    )
    model = SAM2Modified(
        SAM2ImagePredictor(
            build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "./checkpoints/sam2.1_hiera_large.pt",
            )
        )
    )

    x, y = dataset[0]

    masks, _, _ = model(x)

    print(masks.shape)
    fig = __generate_masks_with_points__(model, dataset, 20)
    fig.savefig("batu-simge/modified_sam.png")
