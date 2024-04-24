import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig

from src import utils
import wandb

def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    best_val_acc = 0.0
    best_model = type(model)(opt).cuda() if "cuda" in opt.device else type(model)(opt)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        model.epoch = epoch 
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)
        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels) # push to GPU

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        best_val_acc, best_model = validate_or_test(opt, model, "val", epoch=epoch, best_val_acc=best_val_acc, best_model=best_model)

    print("saving model")
    utils.save_model(best_model)

    return best_model


def validate_or_test(opt, model, partition, epoch=None, best_val_acc=1.0, best_model=None):
    test_time = time.time()
    test_results = defaultdict(float)
    scalar_outputs = {}
    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

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

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)

    if test_results["classification_accuracy"] > best_val_acc:
        best_val_acc = test_results["classification_accuracy"]
        best_model.load_state_dict(model.state_dict())

    model.train()
    return best_val_acc, best_model
@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    run = wandb.init(
    entity="ganFF",
    project = "mlmi",
    name = opt.name, # Wandb creates random run names if you skip this field
    reinit = False, # Allows reinitalizing runs when you re-run this cell
    # run_id = # Insert specific run id here if you want to resume a previous run
    # resume = "must" # You need this to resume previous runs, but comment out reinit = True when using this
    config = dict(opt) ### Wandb Config for your run
    )

    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    
    validate_or_test(opt, model, "test")

    run.finish()

if __name__ == "__main__":
    my_main()
