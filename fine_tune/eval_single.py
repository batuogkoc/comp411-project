import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from datasets import load_dataset

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

def predict_masks(predictor: SAM2ImagePredictor, image):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        # width, height = image.size
        # x_samples = np.linspace(0, 1, 10)
        # y_samples = np.linspace(0, 1, 10)
        # point_coords = np.array(np.meshgrid(x_samples, y_samples)).T.reshape(-1, 2)
        # point_labels = np.ones((point_coords.shape[0],))
        masks, _, _ = predictor.predict(box=np.array([0, 0, 1, 1]))
        # masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)
        return masks

def _predict_masks_test():
    raw_dataset = load_dataset("kowndinya23/Kvasir-SEG")
    train_set = raw_dataset["train"]

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    sample_idxs = [0,1]
    fig, axs = plt.subplots(len(sample_idxs),2+3)
    for i in sample_idxs:
        image = train_set[i]["image"]
        mask_gt = train_set[i]["annotation"]
        # name, image, mask = train_set[i].items()
        axs[i,0].imshow(image)
        axs[i,1].imshow(mask_gt)
        masks = predict_masks(predictor, image)
        print(masks.shape)
        for j, mask in enumerate(masks):
            axs[i,2+j].imshow(masks[j])


    fig.savefig("single.png")

def _generate_masks():
    raw_dataset = load_dataset("kowndinya23/Kvasir-SEG")
    train_set = raw_dataset["train"]

    mask_generator = SAM2AutomaticMaskGenerator(build_sam2(model_cfg, checkpoint))

    sample_idxs = range(10)
    plt.axis("off")
    fig, axs = plt.subplots(len(sample_idxs),2+3)

    for i in sample_idxs:
        image = np.array(train_set[i]["image"].convert("RGB"))
        mask_gt = train_set[i]["annotation"]
        axs[i,0].imshow(image)
        axs[i,1].imshow(mask_gt)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = mask_generator.generate(image)
            for j, mask in enumerate(masks[:3]):
                segmentation = mask["segmentation"]
                axs[i,2+j].imshow(segmentation)

    fig.savefig("eval_single.png")

if __name__ == "__main__":
    _generate_masks()


