# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
import os  # Added
import datetime  # Added
import pandas as pd  # Added
from argparse import ArgumentParser
from typing import Type, List # Ensure List is imported

import torch
import torch.optim as optim
# --- Updated LR Scheduler Import --- 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
# from experiments.lr_scheduler import LinearWarmupCosineAnnealing # Removed step-based scheduler
from torch.optim.lr_scheduler import StepLR # Added StepLR
# ---
from dacite import from_dict
from experiments.data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
# Updated import for Battery Dataset Generator
from experiments.data.battery.battery_dataset import BatteryDatasetGenerator, BatteryDatasetConfig, BatteryDataset
from experiments.data.utils import DataGen
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Import the new Regression Model --- 
from xlstm.xlstm_regression_model import xLSTMRegressionModel, xLSTMRegressionModelConfig
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
    # Use shuffle=True for training loader
    train_loader = DataLoader(train_dataset_gen.train_split, batch_size=cfg.training.batch_size, shuffle=True)
    print("Training DataLoader created.")

    # Load validation datasets using the generator
    val_loaders = {
        key: DataLoader(val_ds, batch_size=cfg.training.batch_size) 
        for key, val_ds in train_dataset_gen.validation_split.items()
    }
    val_loader = val_loaders.get("val") # Get the primary validation loader
    if val_loader is None:
        print("Warning: No validation loader named 'val' found. Early stopping and LR scheduling based on validation loss will not work.")
    print("Validation DataLoaders created.")

    # --- Load testing dataset directly --- 
    print("Loading Testing dataset...")
    test_dataset_cfg_dict = OmegaConf.to_container(cfg.test_dataset.kwargs)
    # Ensure feature_columns is passed correctly based on BatteryDataset constructor
    test_dataset = BatteryDataset(
         file_path=test_dataset_cfg_dict['file_path'], 
         seq_len=test_dataset_cfg_dict['seq_len'],
         pred_len=test_dataset_cfg_dict['pred_len'],
         target_column=test_dataset_cfg_dict['target_column'],
         feature_columns=list(test_dataset_cfg_dict['feature_columns']) # Pass as list
    )
    print("Testing dataset loaded.")

    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    print("Testing DataLoader created.")
    # ---

    # --- Get training config parameters --- 
    pred_len = cfg.dataset.kwargs.pred_len
    if pred_len <= 0:
        raise ValueError("pred_len in dataset config must be positive.")
    device = cfg.training.device
    max_epochs = cfg.training.max_epochs
    patience = cfg.training.early_stopping_patience
    # ---

    # Set up training and validation metrics (obtained from the generator)
    train_metrics = train_dataset_gen.train_metrics.to(device=device)
    val_metrics = train_dataset_gen.validation_metrics.to(device=device)

    # --- Set up model using Regression Model --- 
    print("Initializing Regression Model...")
    model_config = from_dict(xLSTMRegressionModelConfig, OmegaConf.to_container(cfg.model))
    
    # Set dims dynamically if needed
    if model_config.input_dim <= 0:
        print(f"Warning: model.input_dim not set. Using value from dataset: {train_dataset_gen.input_dim}")
        model_config.input_dim = train_dataset_gen.input_dim
    # ... (similar checks for output_dim)
    if model_config.output_dim <= 0:
         print(f"Warning: model.output_dim not set. Using value from dataset: {train_dataset_gen.output_dim}")
         model_config.output_dim = train_dataset_gen.output_dim

    model = xLSTMRegressionModel(model_config).to(device=device)
    print("Regression Model initialized successfully.")

    # --- Calculate and Print Model Parameters --- 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} (~{total_params / 1_000_000:.2f}M)")
    # ---

    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])

    # --- Optimizer setup --- 
    if hasattr(model, '_create_weight_decay_optim_groups'):
        print("Using model's weight decay groups.")
        optim_groups = model._create_weight_decay_optim_groups()
        params_wd = list(optim_groups[0])
        params_no_wd = list(optim_groups[1])
    else:
        print("Using default weight decay grouping.")
        params_wd = list(model.parameters())
        params_no_wd = []
    
    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": params_wd},
            {"weight_decay": 0.0, "params": params_no_wd},
        ),
        lr=cfg.training.lr,
    )
    # ---

    # --- Learning Rate Scheduler --- 
    lr_scheduler_type = cfg.training.get("lr_scheduler_type", "ReduceLROnPlateau").lower()
    print(f"Initializing LR scheduler: {lr_scheduler_type}")

    if lr_scheduler_type == "reducelronplateau":
        lr_scheduler_patience = cfg.training.get("lr_scheduler_patience", 3) 
        print(f"  - ReduceLROnPlateau patience: {lr_scheduler_patience}")
        lr_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=cfg.training.get("lr_factor", 0.1), # Use lr_factor or default
            patience=lr_scheduler_patience,
            verbose=True
        )
    elif lr_scheduler_type == "steplr":
        lr_step_size = cfg.training.get("lr_step_size", 10) # Default step size 10 epochs
        lr_gamma = cfg.training.get("lr_gamma", 0.1) # Default decay factor
        print(f"  - StepLR step_size: {lr_step_size}, gamma: {lr_gamma}")
        lr_scheduler = StepLR(
            optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma,
            verbose=True # Prints message on LR change
        )
    else:
        print(f"Warning: Unknown lr_scheduler_type '{cfg.training.lr_scheduler_type}'. Using no scheduler.")
        lr_scheduler = None # Or potentially default to one type
    # ---

    # --- Determine base device type for autocast --- 
    autocast_device_type = 'cuda' if 'cuda' in device else 'cpu'
    print(f"Using device: {device}, Autocast device type: {autocast_device_type}")
    # ---

    # --- Initialize tracking for best validation loss and early stopping --- 
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(run_dir, 'best_checkpoint.pt')
    print(f"Will save best checkpoint to: {best_ckpt_path}")
    print(f"Early stopping patience: {patience} epochs.")
    # ---

    # --- Epoch-based Training loop --- 
    print(f"Starting training for max {max_epochs} epochs...")
    for epoch in range(1, max_epochs + 1):
        print(f"\n--- Epoch {epoch}/{max_epochs} --- ")
        
        # --- Training Phase --- 
        model.train()
        train_loss_epoch = 0.0
        train_metrics.reset()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, (inputs, labels) in enumerate(train_iterator):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()
            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(inputs)
                relevant_outputs = outputs[:, -pred_len:, :] 
                
                if relevant_outputs.shape != labels.shape:
                     # Handle potential shape mismatch (e.g., unsqueeze labels)
                     if relevant_outputs.shape[-1] == 1 and labels.ndim == relevant_outputs.ndim -1:
                         labels = labels.unsqueeze(-1)
                     else:
                          raise RuntimeError(f"Shape mismatch: Output slice {relevant_outputs.shape}, Labels {labels.shape}")

                loss = nn.functional.mse_loss(relevant_outputs, labels)
            
            if torch.isnan(loss):
                 print(f"WARNING: Loss is NaN at Epoch {epoch}, Batch {batch_idx}. Stopping training.")
                 break # Stop epoch if loss is NaN
            
            loss.backward()
            # Optional: Gradient clipping can be added here if needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_metrics.update(relevant_outputs.detach(), labels)
            train_iterator.set_postfix(loss=loss.item()) # Show loss for current batch

        if torch.isnan(loss):
             break # Stop training completely if NaN occurred
             
        avg_train_loss = train_loss_epoch / len(train_loader)
        computed_train_metrics = train_metrics.compute()
        print(f"Epoch {epoch} Training Complete: Avg Loss={avg_train_loss:.4f}, Metrics={computed_train_metrics}")
        
        # --- Validation Phase --- 
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_metrics.reset()
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
            
            with torch.no_grad():
                for val_inputs, val_labels in val_iterator:
                    val_inputs = val_inputs.to(device=device)
                    val_labels = val_labels.to(device=device)
                    with torch.autocast(
                        device_type=autocast_device_type,
                        dtype=torch_dtype_map[cfg.training.amp_precision],
                        enabled=cfg.training.enable_mixed_precision,
                    ):
                        val_outputs = model(val_inputs)
                        relevant_val_outputs = val_outputs[:, -pred_len:, :]
                        
                        if relevant_val_outputs.shape != val_labels.shape:
                             if relevant_val_outputs.shape[-1] == 1 and val_labels.ndim == relevant_val_outputs.ndim -1:
                                 val_labels = val_labels.unsqueeze(-1)
                             else:
                                  raise RuntimeError(f"Val Shape mismatch: Output slice {relevant_val_outputs.shape}, Labels {val_labels.shape}")
                             
                        v_loss = nn.functional.mse_loss(relevant_val_outputs, val_labels)
                        val_loss += v_loss.item()
                        val_metrics.update(relevant_val_outputs, val_labels)
                        val_iterator.set_postfix(loss=v_loss.item())
                        
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            computed_val_metrics = val_metrics.compute()
            # --- Calculate RMSE from MSE --- 
            val_rmse = torch.sqrt(computed_val_metrics['MeanSquaredError']) if 'MeanSquaredError' in computed_val_metrics else -1.0
            # ---
            print(f"Epoch {epoch} Validation Complete: Avg Loss={avg_val_loss:.4f}, Metrics={computed_val_metrics}, RMSE={val_rmse:.4f}")

            # --- LR Scheduling Step --- 
            if lr_scheduler:
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(avg_val_loss) # Pass metric for ReduceLROnPlateau
                else:
                    lr_scheduler.step() # Other schedulers step without metric

            # --- Early Stopping & Best Model Check --- 
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, best_ckpt_path)
                epochs_no_improve = 0 # Reset patience counter
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve from {best_val_loss:.4f}. Patience: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs with no improvement.")
                    break # Stop training
        else:
             print("Skipping validation, LR scheduling, and early stopping due to no validation loader.")
             # If no validation, maybe save the model from the last epoch?
             # Or rely solely on max_epochs

    print("Training loop finished.")

    # --- Load best checkpoint for testing --- 
    if os.path.exists(best_ckpt_path):
        print(f"\nLoading best model checkpoint from {best_ckpt_path} (Epoch {torch.load(best_ckpt_path)['epoch']}, Val Loss: {best_val_loss:.4f})...")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best checkpoint loaded successfully.")
    else:
        print("\nWarning: No best checkpoint was saved during training. Testing with final model state.")
    # ---

    # --- Testing Phase --- 
    print("\nStarting Test Inference...")
    model.eval() 
    test_predictions = []
    true_targets_list = []
    # --- Initialize test metrics --- 
    test_metrics = train_dataset_gen.validation_metrics.to(device=device)
    test_metrics.reset()
    # ---

    with torch.no_grad():
        for test_inputs, test_targets in tqdm(test_loader, desc="Testing"):
            test_inputs = test_inputs.to(device=device)
            # Keep targets on CPU for list appending, move to device for metric calculation
            targets_cpu = test_targets.cpu()
            targets_device = test_targets.to(device=device)

            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                outputs = model(test_inputs)
                relevant_test_outputs = outputs[:, -pred_len:, :]
            
            # --- Update test metrics --- 
            # Ensure labels have the correct shape if needed before updating metrics
            if relevant_test_outputs.shape != targets_device.shape:
                 if relevant_test_outputs.shape[-1] == 1 and targets_device.ndim == relevant_test_outputs.ndim -1:
                     targets_device = targets_device.unsqueeze(-1)
                 # Add more robust shape checking/handling if necessary

            test_metrics.update(relevant_test_outputs.detach(), targets_device)
            # ---
            
            test_predictions.append(relevant_test_outputs.cpu())
            true_targets_list.append(targets_cpu)

    # Combine predictions and targets
    if not test_predictions:
        print("Warning: No test predictions were generated.")
        return # Exit if testing failed or test set was empty
        
    test_predictions = torch.cat(test_predictions, dim=0)
    true_targets = torch.cat(true_targets_list, dim=0)

    # --- Compute and Print Final Test Metrics --- 
    final_test_metrics = test_metrics.compute()
    # --- Calculate RMSE from MSE --- 
    test_rmse = torch.sqrt(final_test_metrics['MeanSquaredError']) if 'MeanSquaredError' in final_test_metrics else -1.0
    # ---
    print(f"\n--- Test Results --- ")
    print(f"  Test Metrics: {final_test_metrics}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"--------------------")
    # ---

    print("Test Inference Completed.")

    # --- Saving results to CSV --- 
    run_timestamp = os.path.basename(run_dir).replace("run_", "")
    results_filename = f"predictions_xlstm_{run_timestamp}.csv"
    results_path = os.path.join(run_dir, results_filename)
    
    # --- Convert tensors to float32 before converting to numpy ---
    pred_np = test_predictions.float().numpy()
    target_np = true_targets.float().numpy()
    save_raw = False 

    if pred_np.shape[1] == 1 and pred_np.shape[2] == 1:
        pred_final = pred_np.squeeze()
        target_final = target_np.squeeze()
        if pred_final.ndim == 1 and target_final.ndim == 1 and len(pred_final) == len(target_final):
            results_df = pd.DataFrame({'True_Voltage': target_final, 'Predicted_Voltage': pred_final})
            results_df.to_csv(results_path, index=False)
            print(f"Saved single-step predictions to {results_path}")
        else:
             print(f"Warning: Squeezed shapes mismatch. Saving raw.")
             save_raw = True
    elif pred_np.shape[2] == 1:
        num_pred_steps = pred_np.shape[1]
        data_dict = {}
        try:
            for i in range(num_pred_steps):
                data_dict[f'True_Voltage_Step_{i+1}'] = target_np[:, i, 0]
                data_dict[f'Predicted_Voltage_Step_{i+1}'] = pred_np[:, i, 0]
            results_df = pd.DataFrame(data_dict)
            results_df.to_csv(results_path, index=False)
            print(f"Saved multi-step predictions to {results_path}")
        except IndexError:
             print(f"Warning: IndexError during multi-step save. Saving raw.")
             save_raw = True
    else:
        print("Warning: Multi-feature output. Saving raw.")
        save_raw = True

    if save_raw:
         try:
             results_df = pd.DataFrame({'True_Seq': [t.tolist() for t in target_np], 'Predicted_Seq': [p.tolist() for p in pred_np]})
             results_df.to_csv(results_path, index=False)
             print(f"Saved raw prediction sequences to {results_path}")
         except Exception as e:
              print(f"Error saving raw sequences to CSV: {e}")


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
        print(f"ERROR: Config file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load or parse configuration file {args.config}: {e}")
        exit(1)

    print("Starting the main script...")
    main(cfg)
