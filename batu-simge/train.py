import torch
import torchmetrics.segmentation
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
from datetime import datetime
import wandb
import torchmetrics
import os
from matplotlib import pyplot as plt
import numpy as np
from torch_datasets import KvasirSEGDataset
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import albumentations as A
from torch import optim
from train_loop import train


def train_experiment():
    # Constants
    EXPERIMENT_NAME = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_sampled-mask"

    print("-" * 10 + "~TRAIN~" + "-" * 10)

    # Use wandb-core, temporary for wandb's new backend
    wandb.init(
        project="sam2-replication",
        name=EXPERIMENT_NAME,
        config={
            "rng_seed": 42,
            "max_epochs": 50,
            "shuffle_dataloader": False,
            "batch_size": 32,
            "num_workers": 1,
            "dataset": {
                "description": "KvasirSEG from huggingface",
                "image_size_hw": [1024, 1024],
            },
            "model": {
                "checkpoint": "./checkpoints/sam2.1_hiera_large.pt",
                "config_file": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "train_mask_decoder": True,
                "train_prompt_encoder": True,
                "train_image_encoder": False,
                "num_points": 10,
            },
            "learning_rate": 1e-6,
        },
    )
    config = wandb.config

    torch.manual_seed(config["rng_seed"])
    np.random.seed(config["rng_seed"])
    print("Loading datasets...")

    raw_dataset = KvasirSEGDataset(
        "train",
        transform=A.Compose(
            [
                A.Resize(
                    height=config["dataset"]["image_size_hw"][0],
                    width=config["dataset"]["image_size_hw"][1],
                ),
            ]
        ),
    )
    train_set, val_set = random_split(raw_dataset, [0.8, 0.2])

    assert len(train_set) > len(
        val_set
    ), f"Sanity check failed, size of train set ({len(train_set)}) must be greater than size of val set ({len(val_set)})"

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=config["shuffle_dataloader"],
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=config["shuffle_dataloader"],
    )

    print("Setting up model, optim, etc...")
    predictor = SAM2ImagePredictor(
        build_sam2(
            config_file=config["model"]["config_file"],
            ckpt_path=config["model"]["checkpoint"],
        )
    )
    predictor.model.image_encoder.train(config["model"]["train_image_encoder"])
    predictor.model.sam_prompt_encoder.train(config["model"]["train_prompt_encoder"])
    predictor.model.sam_mask_decoder.train(config["model"]["train_mask_decoder"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(predictor.model.parameters(), lr=config["learning_rate"])

    state, metrics = train(
        project_name=None,
        config=None,
        predictor=predictor,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epoch=config["max_epochs"],
        train_set=train_set,
        val_set=val_set,
        experiment_name=EXPERIMENT_NAME,
        printing=True,
        tensorboard_logging=True,
        wandb_logging=True,
        metrics={
            "mIoU": torchmetrics.segmentation.MeanIoU(2, input_format="index")
            # "mae": torchmetrics.MeanAbsoluteError(),
            # "mse": torchmetrics.MeanSquaredError(),
            # "acc": torchmetrics.Accuracy(task="multiclass", num_classes=10)
        },
        num_points=config["model"]["num_points"],
    )


if __name__ == "__main__":
    train_experiment()
