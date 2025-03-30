# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
import os  # Added
import datetime  # Added
import pandas as pd  # Added
from argparse import ArgumentParser
from typing import Type

import torch
import torch.optim as optim
from dacite import from_dict
from experiments.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
# Updated import for Battery Dataset Generator
from experiments.data.battery.battery_dataset import BatteryDatasetGenerator, BatteryDatasetConfig, BatteryDataset
from experiments.data.utils import DataGen
from experiments.lr_scheduler import LinearWarmupCosineAnnealing
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Import the new Regression Model --- 
# from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig # Removed LM model
from xlstm.xlstm_regression_model import xLSTMRegressionModel, xLSTMRegressionModelConfig # Added Regression model
# ---

dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator,
    "battery_dataset_generator": BatteryDatasetGenerator, # Updated key
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_dataset(name, kwargs):
    print(f"Loading dataset generator: {name}")
    cls = dataset_registry[name]
    dataset_gen = cls(from_dict(cls.config_class, OmegaConf.to_container(kwargs)))
    print(f"Dataset generator {name} loaded successfully.")
    return dataset_gen


def main(cfg: DictConfig):
    print("Configuration loaded successfully.")
    # print(OmegaConf.to_yaml(cfg)) # Optionally print full config

    # --- Create Timestamped Run Directory --- 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_base = cfg.training.get("run_dir_base", "training_runs")
    run_dir = os.path.join(run_dir_base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")

    # --- Save Config to Run Directory --- 
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf8') as fp:
        OmegaConf.save(config=cfg, f=fp)
    print(f"Saved configuration to {config_save_path}")
    # ---

    torch.manual_seed(cfg.training.seed)

    # Load training dataset using the generator
    train_dataset_gen = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)
    train_loader = DataLoader(train_dataset_gen.train_split, batch_size=cfg.training.batch_size)
    print("Training DataLoader created.")

    # Load validation datasets using the generator
    val_loaders = {
        key: DataLoader(val_ds, batch_size=cfg.training.batch_size) 
        for key, val_ds in train_dataset_gen.validation_split.items()
    }
    print("Validation DataLoaders created.")

    # --- Load testing dataset directly --- 
    print("Loading Testing dataset...")
    test_dataset_cfg_dict = OmegaConf.to_container(cfg.test_dataset.kwargs)
    test_dataset = BatteryDataset(
         file_path=test_dataset_cfg_dict['file_path'], 
         seq_len=test_dataset_cfg_dict['seq_len'],
         pred_len=test_dataset_cfg_dict['pred_len'],
         target_column=test_dataset_cfg_dict['target_column'],
         feature_columns=tuple(test_dataset_cfg_dict['feature_columns'])
    )
    print("Testing dataset loaded.")

    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    print("Testing DataLoader created.")
    # ---

    # --- Get prediction length from config --- 
    # Needed for slicing model output during loss calculation
    pred_len = cfg.dataset.kwargs.pred_len
    if pred_len <= 0:
        raise ValueError("pred_len in dataset config must be positive.")
    # ---

    # Set up training and validation metrics (obtained from the generator)
    train_metrics = train_dataset_gen.train_metrics.to(device=cfg.training.device)
    val_metrics = train_dataset_gen.validation_metrics.to(device=cfg.training.device)

    # --- Set up model using Regression Model --- 
    print("Initializing Regression Model...")
    # Parse the model config using the new Regression Config class
    model_config = from_dict(xLSTMRegressionModelConfig, OmegaConf.to_container(cfg.model))
    
    # Set input_dim dynamically from dataset generator if not set or invalid in config
    # This makes config less prone to error if feature_columns changes
    if model_config.input_dim <= 0:
        print(f"Warning: model.input_dim not set or invalid in config. Using value from dataset generator: {train_dataset_gen.input_dim}")
        model_config.input_dim = train_dataset_gen.input_dim
    elif model_config.input_dim != train_dataset_gen.input_dim:
         print(f"Warning: model.input_dim ({model_config.input_dim}) differs from dataset generator input_dim ({train_dataset_gen.input_dim}). Using value from config.")

    # Set output_dim based on dataset generator if not set or invalid
    if model_config.output_dim <= 0:
         print(f"Warning: model.output_dim not set or invalid in config. Using value from dataset generator: {train_dataset_gen.output_dim}")
         model_config.output_dim = train_dataset_gen.output_dim
    elif model_config.output_dim != train_dataset_gen.output_dim:
         # This might be intentional if model predicts something different than target dim directly
         print(f"Warning: model.output_dim ({model_config.output_dim}) differs from dataset generator output_dim ({train_dataset_gen.output_dim}). Using value from config.")

    # Instantiate the Regression Model
    model = xLSTMRegressionModel(model_config).to(device=cfg.training.device)
    # model.reset_parameters() # reset_parameters is called within __init__ now
    print("Regression Model initialized successfully.")
    # ---

    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])

    # --- Optimizer setup --- 
    # Check if the model has the specific weight decay method (if using WeightDecayOptimGroupMixin)
    if hasattr(model, '_create_weight_decay_optim_groups'):
        print("Using model's _create_weight_decay_optim_groups for optimizer.")
        optim_groups = model._create_weight_decay_optim_groups()
    else:
        print("Using default parameter grouping for optimizer.")
        # Default: apply weight decay to all parameters
        optim_groups = [model.parameters(), []] # [params_with_wd, params_without_wd]
    
    print("Setting up optimizer and learning rate scheduler...")
    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": list(optim_groups[0])},
            {"weight_decay": 0.0, "params": list(optim_groups[1])},
        ),
        lr=cfg.training.lr,
        # Add betas and eps if needed, e.g.: betas=(0.9, 0.95), eps=1e-8
    )
    # ---

    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )
    print("Optimizer and learning rate scheduler set up successfully.")

    # --- Determine base device type for autocast --- 
    autocast_device_type = 'cuda' if 'cuda' in cfg.training.device else 'cpu'
    print(f"Using device: {cfg.training.device}, Autocast device type: {autocast_device_type}")
    # ---

    # Training loop
    print("Starting training...")
    step = 0
    epoch = 1
    running_loss = 0.0
    save_every_step = cfg.training.get("save_every_step", cfg.training.val_every_step * 5)

    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=len(train_loader), initial=0)
        for inputs, labels in monitoring:
            if step >= cfg.training.num_steps:
                break

            monitoring.set_description_str(f"Steps {step+1}/{cfg.training.num_steps} (Epoch: {epoch}) Loss: {running_loss:.4f}")
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)

            model.train()
            optimizer.zero_grad()
            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(inputs)
                
                # --- Adjust Loss Calculation --- 
                # Model output shape: (batch, seq_len, output_dim)
                # Label shape: (batch, pred_len, output_dim)
                # We need to compare the last `pred_len` steps of the output with the labels.
                relevant_outputs = outputs[:, -pred_len:, :] 
                
                # Check shapes before calculating loss
                if relevant_outputs.shape != labels.shape:
                     print(f"Shape mismatch! Output slice: {relevant_outputs.shape}, Labels: {labels.shape}")
                     # Handle error or adjust logic as needed
                     # For now, let's try to reshape labels if output_dim is 1 and label isn't squeezed
                     if relevant_outputs.shape[-1] == 1 and labels.shape[-1] != 1 and labels.ndim == relevant_outputs.ndim -1 :
                         labels = labels.unsqueeze(-1)
                         print(f"Attempted to fix label shape: {labels.shape}")
                     else:
                          raise RuntimeError(f"Cannot resolve shape mismatch between output slice {relevant_outputs.shape} and labels {labels.shape}")

                loss = nn.functional.mse_loss(relevant_outputs, labels)
                # ---
            
            if torch.isnan(loss):
                 print(f"WARNING: Loss is NaN at step {step+1}. Stopping training.")
                 break
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            current_loss = loss.item()
            if not isinstance(running_loss, (int, float)) or running_loss == 0.0:
                 running_loss = current_loss
            else:
                 running_loss = running_loss * 0.99 + current_loss * 0.01

            step += 1
            # --- Update Metrics --- 
            # Ensure metrics are updated with the same tensors used for loss
            train_metrics.update(relevant_outputs.detach(), labels)
            # ---
            monitoring.set_description_str(f"Steps {step}/{cfg.training.num_steps} (Epoch: {epoch}) Loss: {running_loss:.4f}")


            # Validation loop
            if step % cfg.training.val_every_step == 0:
                val_metrics.reset()
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    val_loader = val_loaders.get("val")
                    if val_loader:
                        for val_inputs, val_labels in val_loader:
                            val_inputs = val_inputs.to(device=cfg.training.device)
                            val_labels = val_labels.to(device=cfg.training.device)
                            with torch.autocast(
                                device_type=autocast_device_type,
                                dtype=torch_dtype_map[cfg.training.amp_precision],
                                enabled=cfg.training.enable_mixed_precision,
                            ):
                                val_outputs = model(val_inputs)
                                # --- Adjust Loss/Metrics for Validation --- 
                                relevant_val_outputs = val_outputs[:, -pred_len:, :]
                                if relevant_val_outputs.shape != val_labels.shape:
                                     # Apply same potential fix as in training
                                     if relevant_val_outputs.shape[-1] == 1 and val_labels.shape[-1] != 1 and val_labels.ndim == relevant_val_outputs.ndim -1:
                                         val_labels = val_labels.unsqueeze(-1)
                                     else:
                                          raise RuntimeError(f"Validation shape mismatch: output slice {relevant_val_outputs.shape}, labels {val_labels.shape}")
                                     
                                v_loss = nn.functional.mse_loss(relevant_val_outputs, val_labels)
                                val_loss += v_loss.item()
                                val_metrics.update(relevant_val_outputs, val_labels)
                                # ---
                        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                        computed_val_metrics = val_metrics.compute()
                        computed_train_metrics = train_metrics.compute()
                        print(
                            f"\n--- Step [{step}/{cfg.training.num_steps}] (Epoch: {epoch}) ---\
"
                            f"  Train Loss (Smoothed): {running_loss:.4f}\
"
                            f"  Train Metrics: {computed_train_metrics}\
"
                            f"  Validation Loss: {avg_val_loss:.4f}\
"
                            f"  Validation Metrics: {computed_val_metrics}\
"
                            f"-----------------------------------"
                        )
                    else:
                        print(f"\nStep [{step}/{cfg.training.num_steps}] (Epoch: {epoch}) - No 'val' loader found.")

                train_metrics.reset()
                model.train()
            
            # Checkpoint Saving
            if step % save_every_step == 0:
                ckpt_path = os.path.join(run_dir, f'checkpoint_step_{step}.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            if step >= cfg.training.num_steps:
                break
        
        if torch.isnan(loss):
             break
             
        epoch += 1

    print("Training completed or stopped.")

    # Save Final Checkpoint
    final_ckpt_path = os.path.join(run_dir, 'checkpoint_final.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
    }, final_ckpt_path)
    print(f"Saved final checkpoint to {final_ckpt_path}")

    # --- Testing after training is complete --- 
    print("\nStarting Test Inference:")
    model.eval()
    test_predictions = []
    true_targets_list = []

    with torch.no_grad():
        for test_inputs, test_targets in tqdm(test_loader, desc="Testing"):
            test_inputs = test_inputs.to(device=cfg.training.device)

            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(test_inputs)
                # --- Get Relevant Test Outputs --- 
                # Take the same slice as used during training/loss calculation
                relevant_test_outputs = outputs[:, -pred_len:, :].cpu()
                # ---
                test_predictions.append(relevant_test_outputs) 
                true_targets_list.append(test_targets.cpu())

    # Combine all predictions and targets
    test_predictions = torch.cat(test_predictions, dim=0)
    true_targets = torch.cat(true_targets_list, dim=0) 

    print("Test Inference Completed.")
    print(f"Number of test predictions generated: {test_predictions.shape[0]}")
    print(f"Number of true targets collected: {true_targets.shape[0]}")
    print(f"Test predictions shape: {test_predictions.shape}") # e.g., (num_samples, pred_len, output_dim)
    print(f"True targets shape: {true_targets.shape}") # e.g., (num_samples, pred_len, output_dim)

    # Saving results to CSV in run_dir
    # --- Modify filename to include model type and timestamp --- 
    run_timestamp = os.path.basename(run_dir).replace("run_", "") # Extract timestamp from folder name
    results_filename = f"predictions_xlstm_{run_timestamp}.csv" 
    results_path = os.path.join(run_dir, results_filename)
    # ---
    pred_np = test_predictions.numpy()
    target_np = true_targets.numpy()

    # Save based on pred_len and output_dim
    if pred_np.shape[1] == 1 and pred_np.shape[2] == 1: # Single step, single feature prediction
        pred_final = pred_np.squeeze()
        target_final = target_np.squeeze()
        if pred_final.ndim == 1 and target_final.ndim == 1 and len(pred_final) == len(target_final):
            results_df = pd.DataFrame({
                'True_Voltage': target_final,
                'Predicted_Voltage': pred_final
            })
            results_df.to_csv(results_path, index=False)
            print(f"Saved single-step predictions to {results_path}")
        else:
             print(f"Warning: Squeezed shapes mismatch (Pred: {pred_final.shape}, Target: {target_final.shape}). Saving raw sequences.")
             save_raw = True
    elif pred_np.shape[2] == 1: # Multi-step, single feature prediction
        num_pred_steps = pred_np.shape[1]
        data_dict = {}
        for i in range(num_pred_steps):
            data_dict[f'True_Voltage_Step_{i+1}'] = target_np[:, i, 0]
            data_dict[f'Predicted_Voltage_Step_{i+1}'] = pred_np[:, i, 0]
        results_df = pd.DataFrame(data_dict)
        results_df.to_csv(results_path, index=False)
        print(f"Saved multi-step predictions to {results_path}")
        save_raw = False
    else: # Multi-feature or other cases - save raw
        print("Warning: Multi-feature output or unexpected shape. Saving raw sequences.")
        save_raw = True

    if save_raw:
         # Fallback: Save raw sequences as lists
         results_df = pd.DataFrame({
             'True_Seq': [t.tolist() for t in target_np],
             'Predicted_Seq': [p.tolist() for p in pred_np]
         })
         results_df.to_csv(results_path, index=False)
         print(f"Saved raw prediction sequences to {results_path}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="experiments/battery_xlstm.yaml")

    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf8") as fp:
            config_yaml = fp.read()
        cfg = OmegaConf.create(config_yaml)
        OmegaConf.resolve(cfg)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load or parse configuration file {args.config}: {e}")
        exit(1)

    print("Starting the main script...")
    main(cfg)
