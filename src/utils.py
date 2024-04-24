import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
import random

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Subset

from src import datasets, models
import wandb

import tonic
from tonic import transforms

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))

    return opt

def get_input_layer_size(opt):
    if opt.input.dataset == "mnist":
        return 784
    elif opt.input.dataset == "nmnist":
        return 1156
    else:
        raise ValueError("Unknown dataset.")

def get_model_and_optimizer(opt):
    model = models.Model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    optimizer = torch.optim.SGD(
        [
            {
                "params": model.parameters(),
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            }
        ]
    )

    return model, optimizer


def get_data(opt, partition):
    dataset = datasets.Dataset(opt, partition, num_classes=10)

    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=20,
        pin_memory=True,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_MNIST_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    if partition in ["train", "val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
        if partition in ["train"]:
            mnist = Subset(mnist, range(50000))
        elif partition in ["val"]:
            mnist = Subset(mnist, range(50000, 60000))
        else:
            raise NotImplementedError
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError
    return mnist


val_dataset_global = None
def get_NMNIST_partition(opt, partition):
    global val_dataset_global
    if partition == "val" and val_dataset_global is not None:
        return val_dataset_global

    sensor_size = tonic.datasets.NMNIST.sensor_size

    torch_transform = lambda x: torch.from_numpy(x).float()

    voxel_transform = transforms.Compose([
        tonic.transforms.ToVoxelGrid(sensor_size=sensor_size,n_time_bins=8),
        torch_transform
    ])
    # Load the NMNIST dataset with the specified transform
    dataset = tonic.datasets.NMNIST(
        save_to=os.path.join(get_original_cwd(), opt.input.path),
        train=partition in ["train", "val"],
        transform=voxel_transform
    )

    if partition in ["train", "val"]:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
        if partition in ["train"]:
            dataset = train_dataset
        elif partition in ["val"]:
            val_dataset_global = val_dataset
            dataset = val_dataset

    return dataset

def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 3):
        return lr * (3 / 2) * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    if opt.model.top_down:
        return optimizer
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )

    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)


def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict

