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
from pprint import pprint


def train_experiment(config=None):
    # Constants
    EXPERIMENT_NAME = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_sampled-mask"

    with wandb.init(config=config, name=EXPERIMENT_NAME, project="sam2-replication"):
        config = wandb.config

        if config["printing"]:
            print("-" * 10 + "~TRAIN~" + "-" * 10)
        torch.manual_seed(config["rng_seed"])
        np.random.seed(config["rng_seed"])
        if config["printing"]:
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

        if config["printing"]:
            print("Setting up model, optim, etc...")
        predictor = SAM2ImagePredictor(
            build_sam2(
                config_file=config["model"]["config_file"],
                ckpt_path=config["model"]["checkpoint"],
            )
        )
        predictor.model.image_encoder.train(config["model"]["train_image_encoder"])
        predictor.model.sam_prompt_encoder.train(
            config["model"]["train_prompt_encoder"]
        )
        predictor.model.sam_mask_decoder.train(config["model"]["train_mask_decoder"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # loss_fn = torch.nn.BCELoss()

        if config["optimizer"]["type"] == "rmsprop":
            optimizer = optim.RMSprop(
                predictor.model.parameters(), lr=config["optimizer"]["learning_rate"]
            )
        elif config["optimizer"]["type"] == "adam":
            optimizer = optim.RMSprop(
                predictor.model.parameters(), lr=config["optimizer"]["learning_rate"]
            )
        else:
            raise AttributeError(
                f"incorrect value for optimizer type: {config['optimizer']['type']}"
            )
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
            printing=config["printing"],
            tensorboard_logging=config["tensorboard_logging"],
            wandb_logging=config["wandb_logging"],
            metrics={
                "mIoU": torchmetrics.segmentation.MeanIoU(2, input_format="index")
                # "mae": torchmetrics.MeanAbsoluteError(),
                # "mse": torchmetrics.MeanSquaredError(),
                # "acc": torchmetrics.Accuracy(task="multiclass", num_classes=10)
            },
            num_points=config["model"]["num_points"],
            num_bg_points=config["model"]["num_bg_points"],
        )


def _regular_training():
    train_experiment(
        config={
            "rng_seed": 42,
            "max_epochs": 50,
            "printing": True,
            "tensorboard_logging": True,
            "wandb_logging": True,
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
                "num_bg_points": 20,
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 1e-7,
            },
        }
    )


def dict_to_wandb_sweep_constant_config_params(input_dict: dict):
    ret = {}
    for key, value in input_dict.items():
        if type(value) is dict:
            value = dict_to_wandb_sweep_constant_config_params(value)
            ret[key] = {"parameters": value}
        else:
            ret[key] = {"value": value}

    return ret


def _sweep():
    const_config = {
        "rng_seed": 42,
        "max_epochs": 5,
        "printing": True,
        "tensorboard_logging": True,
        "wandb_logging": True,
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
            # "num_points": 10,
            # "num_bg_points": 20,
        },
        "optimizer": {
            # "type": "adam",
            # "learning_rate": 1e-7,
        },
    }
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val/mIoU", "goal": "maximize"},
        "parameters": {},
    }
    sweep_config.update(dict_to_wandb_sweep_constant_config_params(const_config))
    sweep_id = wandb.sweep(
        sweep_config,
    )


if __name__ == "__main__":
    _sweep()
