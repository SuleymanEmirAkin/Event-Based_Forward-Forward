dataset = "NMNIST" # MNIST OR NMNIST
configuration = "nmnist_ff_unsupervised_top_down_randomized_frame_switch"

import os
from tqdm import tqdm
import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig

from src import utils
import wandb

def calc(opt, model, partition):
    data_loader = utils.get_data(opt, partition)
    test_results = defaultdict(float)
    scalar_outputs = {}
    num_steps_per_epoch = len(data_loader)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.getcwd()), path, configuration, configuration + "-model.pt")))

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            if not (opt.model.top_down and not opt.training.unsupervised and not opt.training.backpropagation):
                scalar_outputs = model.forward_downstream_classification_model(
                        inputs, labels
                    )

            if not opt.training.unsupervised and not opt.training.backpropagation:
                scalar_outputs = model.forward_downstream_multi_pass(
                    inputs, labels, scalar_outputs=scalar_outputs
                )

            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    return test_results


path = os.path.join("configs_models", dataset)

conf_path = os.path.join(path, configuration)
@hydra.main(config_path=conf_path, config_name='config')
def validate(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    start_time = time.time()
    train_accuracy = calc(opt, model, "train")
    # val_accuracy = calc(opt, model, "val")
    test_accuracy = calc(opt, model, "test")
    with open(os.path.join(os.path.dirname(os.getcwd()), path, configuration, "results.txt"), 'w') as file:
        file.write(f"Train Accuracy: {train_accuracy['classification_accuracy']}\n")
        # file.write(f"Val Accuracy: {val_accuracy['classification_accuracy']}\n")
        file.write(f"Test Accuracy: {test_accuracy['classification_accuracy']}\n")
        if "classification_accuracy_3_5" in train_accuracy:
            file.write(f"-----------------------------------------------------------\n")
            file.write(f"Train Accuracy_3_5: {train_accuracy['classification_accuracy_3_5']}\n")
            # file.write(f"Val Accuracy_3_5: {val_accuracy['classification_accuracy_3_5']}\n")
            file.write(f"Test Accuracy_3_5: {test_accuracy['classification_accuracy_3_5']}\n")
        if "multi_pass_classification_accuracy" in train_accuracy:
            file.write(f"-----------------------------------------------------------\n")
            file.write(f"Multi Pass Train Accuracy: {train_accuracy['multi_pass_classification_accuracy']}\n")
            # file.write(f"Multi Pass Val Accuracy: {val_accuracy['multi_pass_classification_accuracy']}\n")
            file.write(f"Multi Pass Test Accuracy: {test_accuracy['multi_pass_classification_accuracy']}\n")
validate()