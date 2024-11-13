import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from train_helpers import train
import torchmetrics
from lossless_fir_layer import LinearLosslessFIRLayer
from lossless_fir_layer import check_losslessness
import wandb
from datetime import datetime


class DelayDataset(Dataset):
    def __init__(self, sample_count, sample_length, num_channels, delay_amount=1, same=False):
        super().__init__()
        # self.sample_count = sample_count
        # self.sample_length = sample_length
        # self.delay_amount = delay_amount
        if same:
            self.data = torch.broadcast_to(torch.rand((1, sample_length, num_channels)), (sample_count, sample_length, num_channels))
        else:
            self.data = torch.rand((sample_count, sample_length, num_channels))
        self.delay_amount = delay_amount

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        L, C = x.shape
        y = torch.cat([torch.zeros((self.delay_amount, C)), x[:-self.delay_amount]], dim=0)

        return x, y

def train_sweep(config=None):
    with wandb.init(project="linear-lossless-fir", config=config):
        config = wandb.config
        config.update(
            {
                "max_epochs": 20,
                "shuffle_dataloader": True,
                "dataset": "delayed uniform unit random synthetic dataset",
                "train_set":{
                    "sample_num": 8000,
                    "sequence_len": 800,
                    "channels": config["model_channel_num"],
                    "identical_samples": False,
                },
                "val_set":{
                    "sample_num": 2000,
                    "sequence_len": 800,
                    "channels": config["model_channel_num"],
                    "identical_samples": False,
                },
                "learning_rate": 0.001,
                "model_type": "cascade of linear lossless fir layers",
            }
        )

        # print("-"*10 + "~TRAIN~" + "-"*10)
        # print(f"RECORD: {RECORD}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        

        # print(f"Using device: {device.type}")

        # print("Loading datasets...")


        # Dataset and transform
        
        train_set = DelayDataset(config["train_set"]["sample_num"], config["train_set"]["sequence_len"], config["train_set"]["channels"], same=config["train_set"]["identical_samples"])
        val_set = DelayDataset(config["val_set"]["sample_num"], config["val_set"]["sequence_len"], config["val_set"]["channels"], same=config["val_set"]["identical_samples"])



        assert len(train_set) > len(val_set), f"Sanity check failed, size of train set ({len(train_set)}) must be greater than size of val set ({len(val_set)})"

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=config["shuffle_dataloader"])
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=config["shuffle_dataloader"])


        print("Setting up model, optim, etc...")


        # model = nn.Sequential(*[LinearLosslessFIRLayer(config["model_per_layer_segment_num"], train_set[0][0].shape[-1]) for i in range(config["model_layer_num"])])
        model = LinearLosslessFIRLayer(config["model_per_layer_segment_num"], train_set[0][0].shape[-1])
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        scheduler = None

        train(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            loss_fn=loss_fn,
            num_epoch=config["max_epochs"], 
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device, 
            printing=True,
            recording=False,
            wandb_logging=True,
            experiment_name=EXPERIMENT_NAME,
            metrics={
                # "mae": torchmetrics.MeanAbsoluteError(),
                "mse": torchmetrics.MeanSquaredError()
                # "acc" : torchmetrics.Accuracy(task="multiclass", num_classes=18)
            },
            checkpoint_folder=None,
            load_progress_path=LOAD_PROGRESS_PATH)


if __name__ == "__main__":
    EXPERIMENT_NAME = "delay-learn_"+datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    BATCH_SIZE = 32
    NUM_WORKERS = 1
    RECORD=False
    LOAD_PROGRESS_PATH = None
    torch.manual_seed(42)


    # Use wandb-core, temporary for wandb's new backend  
    wandb.require("core")
    wandb.login()

    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "val/mse"},
        "parameters": {
            "model_layer_num": {"values": [1, 2, 4]},
            "model_per_layer_segment_num": {"values": [1, 2, 4]},
            "model_channel_num": {"values": [1, 2, 4, 8]},            
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="linear-lossless-fir")

    wandb.agent(sweep_id=sweep_id, function=train_sweep, count=64)
    # wandb.init(
    #     project="linear-lossless-fir",
    #     name=EXPERIMENT_NAME,
    #     config={
    #         "max_epochs": 20,
    #         "shuffle_dataloader": True,
    #         "dataset": "delayed uniform unit random synthetic dataset",
    #         "train_set":{
    #             "sample_num": 8000,
    #             "sequence_len": 800,
    #             "channels": 2,
    #             "identical_samples": False,
    #         },
    #         "val_set":{
    #             "sample_num": 2000,
    #             "sequence_len": 800,
    #             "channels": 2,
    #             "identical_samples": False,
    #         },
    #         "model": {
    #             "type": "cascade of linear lossless fir layers",
    #             "layer_num": 1,
    #             "per_layer_segment_num":4,
    #         },
    #         "learning_rate": 1e-3,
    #     }
    # )
    # config = wandb.config

    # print("-"*10 + "~TRAIN~" + "-"*10)
    # print(f"RECORD: {RECORD}")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    

    # print(f"Using device: {device.type}")

    # print("Loading datasets...")


    # # Dataset and transform
    # raw_dataset = DelayDataset(10000, 800, 8)
    # train_set, val_set = torch.utils.data.random_split(raw_dataset, [0.8, 0.2])
    
    # # train_set = DelayDataset(config["train_set"]["sample_num"], config["train_set"]["sequence_len"], config["train_set"]["channels"], same=config["train_set"]["identical_samples"])
    # # val_set = DelayDataset(config["val_set"]["sample_num"], config["val_set"]["sequence_len"], config["val_set"]["channels"], same=config["val_set"]["identical_samples"])



    # assert len(train_set) > len(val_set), f"Sanity check failed, size of train set ({len(train_set)}) must be greater than size of val set ({len(val_set)})"

    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=config["shuffle_dataloader"])
    # val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=config["shuffle_dataloader"])

    # for i, (x, y) in enumerate(train_loader):
    #     # print(i)
    #     print(x.shape)
    #     print(y.shape)
    #     print(x[0, :10, 0])
    #     print(y[0, :10, 0])
    #     if i == 0:
    #         break

    # print("Setting up model, optim, etc...")


    # # model = nn.Sequential(*[LinearLosslessFIRLayer(config["model"]["per_layer_segment_num"], train_set[0][0].shape[-1]) for i in range(config["model"]["layer_num"])])
    # # model = nn.Sequential(*[LinearLosslessFIRLayer(1, train_set[0][0].shape[-1]) for i in range(1)])
    # model = LinearLosslessFIRLayer(1, train_set[0][0].shape[-1])
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # scheduler = None




    # train(
    #     model=model, 
    #     optimizer=optimizer, 
    #     scheduler=scheduler, 
    #     loss_fn=loss_fn,
    #     num_epoch=config["max_epochs"], 
    #     train_loader=train_loader, 
    #     val_loader=val_loader,
    #     device=device, 
    #     printing=True,
    #     recording=RECORD,
    #     wandb_logging=False,
    #     experiment_name=EXPERIMENT_NAME,
    #     metrics={
    #         # "mae": torchmetrics.MeanAbsoluteError(),
    #         "mse": torchmetrics.MeanSquaredError()
    #         # "acc" : torchmetrics.Accuracy(task="multiclass", num_classes=18)
    #     },
    #     checkpoint_folder=None,
    #     load_progress_path=LOAD_PROGRESS_PATH)

    # data = torch.rand((1, 20, train_set[0][0].shape[-1]), device=device)
    # # print(model.are_constraints_satisfied())
    # # print(check_losslessness(data, model.V, model.Theta))
    # table_data = [
    #     data[0, :, 0],
    #     model(data)[0, :, 0],
    # ]
    # table = wandb.Table(rows=["input", "model output"], data=table_data)
    # wandb.log({"prediction": table})
    wandb.finish()