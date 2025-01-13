from torch.utils.data import Dataset
from typing import Literal
from datasets import load_dataset
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class KvasirSEGDataset(Dataset):
    def __init__(
        self, split: Literal["train", "validation"], transform: A.DualTransform = None
    ):
        super().__init__()
        self.dataset = load_dataset("kowndinya23/Kvasir-SEG")[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform:
            transformed_sample = self.transform(
                image=np.array(sample["image"]), mask=np.array(sample["annotation"])
            )
            return transformed_sample["image"], transformed_sample["mask"]
        else:
            return sample["image"], sample["annotation"]


if __name__ == "__main__":
    dset = KvasirSEGDataset("train")
    dset_transformed = KvasirSEGDataset(
        "train",
        transform=A.Compose([A.CenterCrop(100, 100), ToTensorV2()]),
    )
    x, y = dset[0]
    x_t, y_t = dset_transformed[0]
    print(x_t.shape, y_t.shape)

    fig, axs = plt.subplots(2, 2)
    plt.axis("off")
    axs[0, 0].imshow(x)
    axs[0, 1].imshow(y)

    axs[1, 0].imshow(x_t.permute(1, 2, 0))
    axs[1, 1].imshow(y_t)

    plt.savefig("temp.png")
