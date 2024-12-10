import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = Image.open("test_code/polyp.jpg")
    width, height = image.size
    print(image.size)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(box=np.array([0,0, width, height]))
    image = Image.fromarray(masks)
    image.save("output_image.jpg", "JPEG")

