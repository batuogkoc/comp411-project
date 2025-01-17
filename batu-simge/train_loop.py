from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchmetrics
import wandb
import os
import time
import numpy as np
import torch
from datasets import *
from train_helpers import (
    compute_metrics,
    log_metrics_to_wandb,
    log_metrics,
    InplacePrinter,
    RunningAverageLogger,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch import nn
from torch import optim
from mask_from_y import extract_points_from_mask, generate_masks_with_points


# def IoU(mask_pred, mask):
#     intersection = torch.logical_and


def train(
    project_name: str,
    config: dict,
    predictor: SAM2ImagePredictor,
    optimizer: optim.Optimizer,
    loss_fn,
    num_epoch: int,
    train_set: DataLoader,
    val_set: DataLoader,
    experiment_name: None | str = None,
    printing: bool = True,
    tensorboard_logging: bool = True,
    wandb_logging: bool = True,
    metrics: dict[str, torchmetrics.Metric] = {},
    num_points: int = 3,
):
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    start_epoch = 0

    tensorboard_writer = (
        SummaryWriter(os.path.join("runs_tensorboard", str(experiment_name)))
        if tensorboard_logging
        else None
    )
    running_average_training_loss_logger = RunningAverageLogger()
    val_loss_logger = RunningAverageLogger()

    printer = InplacePrinter(2 + len(metrics))

    for epoch in range(start_epoch, num_epoch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if printing:
                printer.reset()
                print("-" * 5 + f"EPOCH: {epoch}" + "-" * 5)
            start = time.time()

            running_average_training_loss_logger.reset()

            for metric in metrics.values():
                metric.reset()

            for i, (x, y) in enumerate(train_set):
                y = (torch.tensor(y) > 0).to(predictor.model.device).to(torch.float32)
                point_coords = extract_points_from_mask(y, num_points=num_points)
                point_labels = np.ones(point_coords.shape[0])
                predictor.set_image(x)
                # mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(point_coords=point_coords, point_labels=point_labels, normalize_coords=True, box=None, mask_logits=None)
                # sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

                y_pred, _, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    return_logits=True,
                )
                y_pred = y_pred[0]

                mask = (y > 0).to(torch.long).cpu()
                mask_pred = (y_pred > predictor.mask_threshold).to(torch.long).cpu()
                loss = loss_fn(y_pred, y)

                predictor.model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_metric_values = {}
                for metric_name, metric in metrics.items():
                    epoch_metric_values[metric_name] = metric(mask_pred, mask).item()

                running_average_training_loss_logger.add_value(loss.item())

                log_metrics(
                    None,
                    global_step=epoch * len(train_set) + i,
                    prefix="within_epoch_train/",
                    extra={
                        "epoch": epoch,
                        "loss": loss.item(),
                        "ratl": running_average_training_loss_logger.get_avg(),
                        **epoch_metric_values,
                    },
                    log_wandb=False,
                    tensorboard_writer=tensorboard_writer,
                )
                if i % 1 == 0 and i != 0:
                    fraction_done = max(i / len(train_set), 1e-6)
                    time_taken = time.time() - start
                    if printing:
                        for metric_name, metric in metrics.items():
                            printer.print(
                                f"{metric_name} : {epoch_metric_values[metric_name]}"
                            )
                        printer.print(
                            f"e: {epoch} | i: {i} | loss: {loss.item():2.3f} | ratl: {running_average_training_loss_logger.get_avg():2.3f}"
                        )
                        printer.print(
                            f"{fraction_done*100:2.2f}% | est time left: {time_taken*(1-fraction_done)/fraction_done:.1f} s | est total: {time_taken/fraction_done:.1f} s"
                        )

            total_time_taken = time.time() - start

            log_metrics(
                metrics,
                global_step=epoch,
                prefix="train/",
                extra={
                    "epoch": epoch,
                    "loss": running_average_training_loss_logger.get_avg(),
                    "total_time": total_time_taken,
                    "per_epoch_time": total_time_taken / len(train_set),
                },
                log_wandb=wandb_logging,
                tensorboard_writer=tensorboard_writer,
            )
            if printing:
                print("--Train--")
                for metric_name, metric in metrics.items():
                    print(f"{metric_name} : {metric.compute()}")
            for metric_name, metric in metrics.items():
                metric.reset()

            val_loss_logger.reset()

            with torch.inference_mode():
                for i, (x, y) in enumerate(val_set):
                    y = (
                        (torch.tensor(y) > 0)
                        .to(predictor.model.device)
                        .to(torch.float32)
                    )
                    point_coords = extract_points_from_mask(y, num_points=num_points)
                    point_labels = np.ones(point_coords.shape[0])
                    predictor.set_image(x)
                    # mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(point_coords=point_coords, point_labels=point_labels, normalize_coords=True, box=None, mask_logits=None)
                    # sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

                    y_pred, _, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        return_logits=True,
                    )
                    y_pred = y_pred[0]
                    mask = (y > 0).to(torch.long).cpu()
                    mask_pred = (y_pred > predictor.mask_threshold).to(torch.long).cpu()

                    loss = loss_fn(y_pred, y)
                    val_loss_logger.add_value(loss.item())

                    for metric_name, metric in metrics.items():
                        metric(mask_pred, mask)

            if printing:
                print("--Val--")
                for metric_name, metric in metrics.items():
                    print(f"{metric_name} : {metric.compute().item()}")
                print(
                    f"train loss: {running_average_training_loss_logger.get_avg()} | val loss: {val_loss_logger.get_avg()}"
                )
            if wandb_logging:
                wandb.log(
                    {
                        "predict_vis": wandb.Image(
                            generate_masks_with_points(
                                predictor, val_set, num_points=num_points
                            )
                        )
                    }
                )
            log_metrics(
                metrics,
                global_step=epoch,
                prefix="val/",
                extra={
                    "epoch": epoch,
                    "loss": val_loss_logger.get_avg(),
                },
                log_wandb=wandb_logging,
                tensorboard_writer=tensorboard_writer,
            )

    return metrics
