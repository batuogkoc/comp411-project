import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch.utils.data import DataLoader
import torchmetrics
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from py_utils import *
import wandb
def compute_metrics(metrics):
    return {metric_name:metric.compute().item() for metric_name, metric in metrics.items()}

def log_metrics_to_wandb(metrics, prefix="", extra=None):
    metric_values = compute_metrics(metrics)
    to_log = {prefix + metric_name: metric_value for metric_name, metric_value in metric_values.items()}
    if extra:
        to_log = {**to_log, **extra}
    wandb.log(to_log)

def train(
        model:nn.Module, 
        optimizer:torch.optim.Optimizer, 
        scheduler:torch.optim.lr_scheduler.LRScheduler, 
        loss_fn,
        num_epoch:int, 
        train_loader:DataLoader, 
        val_loader:DataLoader,
        device:torch.device, 
        experiment_name:None|str=None,
        printing:bool=True,
        recording:bool=True,
        wandb_logging:bool=True,
        metrics:dict[str, torchmetrics.Metric]={},
        checkpoint_folder:None|str=None,
        load_progress_path:None|str=None):
    start_epoch = 0
    if load_progress_path:
        print("Loading checkpoint...")
        checkpoint_folder, _ = os.path.split(load_progress_path)
        path_components = os.path.normpath(load_progress_path).split(os.sep)
        experiment_name = path_components[1]
        state = torch.load(load_progress_path, map_location=device)
        if "epoch" in state and not "epoch_progress" in state:
            start_epoch = state["epoch"] + 1
        else:
            assert False, "Must start progress from a finished epoch"
        
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optim_state_dict"])
        if state["scheduler_state_dict"]:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        print(state.keys())
    else:
        print("Starting fresh run...")
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if checkpoint_folder is None:
            checkpoint_folder = f"runs/{experiment_name}"

    if recording:
        # tensorboard_writer = SummaryWriter(f'runs_tensorboard\\{experiment_name}')
        os.makedirs(checkpoint_folder, exist_ok=True)
    else:
        # shutil.rmtree(f'runs_tensorboard\\temp')
        # tensorboard_writer = SummaryWriter(os.path.join("runs_tensorboard", str(experiment_name)))
        checkpoint_folder = None

    running_average_training_loss_logger = RunningAverageLogger()
    val_loss_logger = RunningAverageLogger()

    printer = InplacePrinter(2+len(metrics))
    
    model.to(device)

    for metric in metrics.values():
        metric.to(device)
    
    for epoch in range(start_epoch, num_epoch):
        if printing:
            printer.reset()
            print("-"*5 + f"EPOCH: {epoch}" + "-"*5)
        start = time.time()

        running_average_training_loss_logger.reset()

        for metric in metrics.values():
            metric.reset()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            epoch_metric_values = {}
            for metric_name, metric in metrics.items():
                epoch_metric_values[metric_name] = metric(y_pred, y).item()

            loss.backward()
            optimizer.step()
            
            running_average_training_loss_logger.add_value(loss.item())

            if wandb_logging:
                wandb.log({"within_epoch_train/"+metric_name:metric_value for metric_name, metric_value in epoch_metric_values.items()})

            if i % 10 == 0 and i != 0:
                fraction_done = max(i/len(train_loader), 1e-6)
                time_taken = (time.time()-start)
                if printing:
                    for metric_name, metric in metrics.items():
                        printer.print(f"{metric_name} : {epoch_metric_values[metric_name]}")
                    printer.print(f"e: {epoch} | i: {i} | loss: {loss.item():2.3f} | ratl: {running_average_training_loss_logger.get_avg():2.3f}")
                    printer.print(f"{fraction_done*100:2.2f}% | est time left: {time_taken*(1-fraction_done)/fraction_done:.1f} s | est total: {time_taken/fraction_done:.1f} s")
                # if tensorboard_writer:
                #     tensorboard_writer.add_scalar("running_average_training_loss", running_average_training_loss_logger.get_avg(), epoch*len(train_loader) + i)
                #     for metric_name, metric in metrics.items():
                #         tensorboard_writer.add_scalar(f"train_{metric_name}", epoch_metric_values[metric_name], epoch*len(train_loader) + i)

            if checkpoint_folder and i%10000 == 0 and i!=0:
                torch.save({
                    "epoch": epoch,
                    "epoch_progress": i,
                    "epoch_size": len(train_loader),
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "running_average_training_loss": running_average_training_loss_logger.get_avg(),
                    "epoch_metric_values": epoch_metric_values,
                }, os.path.join(checkpoint_folder, f"e-{epoch}-i-{i}-mbtl-{loss.item()}.pt"))

        if wandb_logging:
            log_metrics_to_wandb(metrics,
                            prefix="train/",
                            extra={
                                "epoch": epoch,
                                "loss": running_average_training_loss_logger.get_avg(),
                            })
        if printing:
            print("--Train--")
            for metric_name, metric in metrics.items():
                print(f"{metric_name} : {metric.compute()}")
                metric.reset()

        val_loss_logger.reset()

        with torch.inference_mode():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)

                model.eval()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss_logger.add_value(loss.item())

                for metric_name, metric in metrics.items():
                    metric(y_pred, y)


        if printing:
            print("--Val--")
            for metric_name, metric in metrics.items():
                print(f"{metric_name} : {metric.compute().item()}")
            print(f"train loss: {running_average_training_loss_logger.get_avg()} | val loss: {val_loss_logger.get_avg()}")
        log_metrics_to_wandb(metrics,
            prefix="val/",
            extra={
                "epoch": epoch,
                "loss": val_loss_logger.get_avg(),
            })
        if scheduler:
            scheduler.step()

        # if tensorboard_writer:
        #     tensorboard_writer.add_scalar("train_loss", running_average_training_loss_logger.get_avg(), epoch)
        #     tensorboard_writer.add_scalar("test_loss", val_loss_logger.get_avg(), epoch)
        #     for metric_name, metric in metrics.items():
        #         tensorboard_writer.add_scalar(f"val_{metric_name}", metric.compute().item(), epoch)

        if checkpoint_folder:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "training_loss": running_average_training_loss_logger.get_avg(),
                "val_loss": val_loss_logger.get_avg(),
                "metrics": compute_metrics(),
            }, os.path.join(checkpoint_folder, f"e-{epoch}-train_l-{running_average_training_loss_logger.get_avg()}-test_l-{val_loss_logger.get_avg()}.pt"))

        
    # if wandb_logging:
    #     for metric_name, metric_val in compute_metrics(metrics).items(): 
    #         wandb.summary[metric_name] = metric_val 
    
    return model, metrics



# if __name__ == "__main__":
#     RECORD=False
#     print("-"*10 + "~TRAIN~" + "-"*10)
#     print(f"RECORD: {RECORD}")
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
        

#     print(f"Using device: {device.type}")

#     print("Loading datasets...")

#     raw_dataset = torchtext.datasets.AG_NEWS(split="train")
#     train_set, val_set = torch.utils.data.random_split(raw_dataset, [0.8, 0.2])

#     assert len(train_set) > len(val_set), f"Sanity check failed, size of train set ({len(train_set)}) must be greater than size of val set ({len(val_set)})"
#     NUM_WORKERS = 1
#     SHUFFLE = True
#     BATCH_SIZE=16
#     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
#     val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)

#     print("Setting up model, optim, etc...")

#     model = None


#     loss_fn = nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(model.parameters())
#     # optimizer = torch.optim.Adam(model.parameters(), lr=0.000003)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.3)
#     scheduler = None
    
#     # LOAD_PROGRESS_PATH = "runs/2024-06-20T15:23:33/e-8-train_l-813.9820446734548-test_loss-869.9771018757278.pt"

#     LOAD_PROGRESS_PATH = None
    


#     train(
#         model=model, 
#         optimizer=optimizer, 
#         scheduler=scheduler, 
#         loss_fn=loss_fn,
#         num_epoch=50, 
#         train_loader=train_loader, 
#         val_loader=val_loader,
#         device=device, 
#         printing=True,
#         recording=RECORD,
#         metrics={
#             # "mae": torchmetrics.MeanAbsoluteError(),
#             # "mse": torchmetrics.MeanSquaredError()
#             "acc" : torchmetrics.Accuracy(task="multiclass", num_classes=18)
#         },
#         checkpoint_folder=None,
#         load_progress_path=LOAD_PROGRESS_PATH)