import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from torch.utils.data import DataLoader
import torchmetrics
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb


class RunningAverageLogger:
    def __init__(self, initial_value=0):
        self.initial_value = initial_value
        self.reset()

    def reset(self):
        self._count = 0
        self._running_sum = self.initial_value

    def get_avg(self):
        if self._count == 0:
            return self._running_sum
        return float(self._running_sum) / float(self._count)

    def add_value(self, value):
        self._running_sum += float(value)
        self._count += 1

    def count(self):
        return self._count


class InplacePrinter:
    CURSOR_UP = "\033[F"
    CLEAR_TILL_LINE_END = "\033[K"

    def __init__(self, max_lines, auto_clear=True):
        self.max_lines = max_lines
        self.auto_clear = True
        self._curr_num_lines = 0

    def reset(self):
        self._curr_num_lines = 0

    def clear(self):
        for i in range(self._curr_num_lines):
            print(self.CURSOR_UP + "\r" + self.CLEAR_TILL_LINE_END, end="", flush=True)
        self._curr_num_lines = 0

    def print(self, str: str):
        if self.max_lines != -1 and self._curr_num_lines == self.max_lines:
            if self.auto_clear:
                self.clear()
            else:
                return

        print(str, flush=True)
        self._curr_num_lines += 1


def compute_metrics(metrics):
    return {
        metric_name: metric.compute().item() for metric_name, metric in metrics.items()
    }


def log_metrics_to_wandb(metrics, prefix="", extra=None):
    metric_values = compute_metrics(metrics)
    to_log = {
        prefix + metric_name: metric_value
        for metric_name, metric_value in metric_values.items()
    }
    if extra:
        to_log = {**to_log, **extra}
    wandb.log(to_log)


def log_metrics(
    metrics,
    global_step,
    prefix="",
    extra=None,
    log_wandb=True,
    tensorboard_writer: SummaryWriter = None,
):
    if metrics:
        metric_values = compute_metrics(metrics)
        to_log = {
            prefix + metric_name: metric_value
            for metric_name, metric_value in metric_values.items()
        }
    else:
        to_log = {}
    if extra:
        to_log = {**to_log, **{(prefix + name): value for name, value in extra.items()}}
    if log_wandb:
        wandb.log(to_log, step=global_step)

    if tensorboard_writer:
        for label, value in to_log.items():
            tensorboard_writer.add_scalar(label, value, global_step)
