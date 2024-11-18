# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
from argparse import ArgumentParser
from typing import Type

import torch
import torch.optim as optim
from dacite import from_dict
from experiments.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
from experiments.data.battery.battery_dataset import BatteryDataset
from experiments.data.utils import DataGen
from experiments.lr_scheduler import LinearWarmupCosineAnnealing
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator,
    "battery_dataset": BatteryDataset,
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_dataset(name, kwargs):
    print(f"Loading dataset: {name}")
    cls = dataset_registry[name]
    dataset = cls(from_dict(cls.config_class, OmegaConf.to_container(kwargs)))
    print(f"Dataset {name} loaded successfully.")
    return dataset


def main(cfg: DictConfig):
    print("Configuration loaded successfully.")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)

    # Load training dataset
    train_dataset = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)
    train_loader = DataLoader(train_dataset.train_split, batch_size=cfg.training.batch_size)
    print("Training dataset loaded and DataLoader created.")

    # Load validation datasets (if available)
    val_loaders = {
        key: DataLoader(val_ds, batch_size=cfg.training.batch_size) for key, val_ds in train_dataset.validation_split.items()
    }
    print("Validation DataLoaders created.")

    # Load testing dataset for inference
    test_dataset = load_dataset(cfg.test_dataset.name, cfg.test_dataset.kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    print("Testing dataset loaded and DataLoader created.")

    # Set up training and validation metrics
    train_metrics = train_dataset.train_metrics.to(device=cfg.training.device)
    val_metrics = train_dataset.validation_metrics.to(device=cfg.training.device)

    # Set up model
    print("Initializing model...")
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
        device=cfg.training.device
    )
    model.reset_parameters()
    print("Model initialized successfully.")

    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])

    optim_groups = model._create_weight_decay_optim_groups()

    # Optimizer and learning rate scheduler setup
    print("Setting up optimizer and learning rate scheduler...")
    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=cfg.training.lr,
    )
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )
    print("Optimizer and learning rate scheduler set up successfully.")

    # Training loop
    print("Starting training...")
    step = 0
    epoch = 1
    running_loss = 0.0
    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=0, initial=0)
        for inputs, labels in monitoring:
            monitoring.set_description_str(f"Steps {step+1}/{cfg.training.num_steps} (Epoch: {epoch})")
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)

            model.train()
            optimizer.zero_grad()
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(inputs.to(device=cfg.training.device))
                loss = nn.functional.mse_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                running_loss = running_loss * step / (step + 1) + loss.item() * 1 / (step + 1)

            step += 1
            train_metrics.update(outputs, labels)
            print(f"Step {step}: Loss = {loss.item():.4f}")

            if step % cfg.training.val_every_step == 0:
                print(
                    f"\nStep [{step+1}/{cfg.training.num_steps}] (Epoch: {epoch}), Loss: {running_loss:.4f},"
                    f" Metrics: {train_metrics.compute()}"
                )
                # Validation loop
                for vl_name, val_loader in val_loaders.items():
                    model.eval()
                    val_loss = 0.0
                    val_metrics.reset()
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs = val_inputs.to(device=cfg.training.device)
                            val_labels = val_labels.to(device=cfg.training.device)
                            with torch.autocast(
                                device_type=cfg.training.device,
                                dtype=torch_dtype_map[cfg.training.amp_precision],
                                enabled=cfg.training.enable_mixed_precision,
                            ):
                                val_outputs = model(val_inputs)
                                loss = nn.functional.mse_loss(val_outputs, val_labels)
                                val_loss += loss.item()
                                val_metrics.update(val_outputs, val_labels)
                        print(
                            f"Validation[{vl_name}] Loss: {val_loss/len(val_loader):.4f},"
                            f" Metrics: {val_metrics.compute()}"
                        )

            if step >= cfg.training.num_steps:
                break
        epoch += 1
    print("Training completed.")

    # Testing after training is complete
    print("\nStarting Test Inference:")
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for test_inputs, _ in tqdm(test_loader):
            test_inputs = test_inputs.to(device=cfg.training.device)
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(test_inputs)
                test_predictions.append(outputs.cpu())

    # Combine all predictions
    test_predictions = torch.cat(test_predictions, dim=0)
    print("Test Inference Completed.")
    print(f"Number of test predictions: {test_predictions.shape[0]}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="experiments/battery_xlstm.yaml")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    print("Starting the main script...")
    main(cfg)
