import pandas as pd
import torch
import torchmetrics
from torch.utils.data import Dataset, Subset, random_split
from typing import Mapping, Tuple, Any
from dataclasses import dataclass
from ..utils import DataGen
from ...metrics import SequenceAccuracy
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# Configuration class for Battery Dataset
@dataclass
class BatteryDatasetConfig:
    train_file_path: str
    test_file_path: str
    seq_len: int
    pred_len: int
    target_column: str
    feature_columns: list[str] = ('Current', 'Temp', 'SOC')
    val_split_percent: float = 0.15

# BatteryDataset class definition
class BatteryDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        pred_len: int,
        target_column: str,
        feature_columns: Tuple[str, ...],
    ):
        try:
            self.data = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {e}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_column = target_column
        self.feature_columns = feature_columns

        # Validate columns
        for col in list(feature_columns) + [target_column]:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset at {file_path}")

        # Extract features and target
        self.features = self.data[list(self.feature_columns)].values
        self.target = self.data[self.target_column].values

        self.X, self.y = self.create_sequences(self.features, self.target)
        self.input_dim = self.features.shape[1]
        self.output_dim = 1 # Predicting a single value (Voltage)

    def create_sequences(self, features, target):
        X, y = [], []
        # Adjust loop range to ensure indices are valid
        max_start_index = len(features) - self.seq_len - self.pred_len
        if max_start_index < 0:
             raise ValueError(f"Dataset too short ({len(features)} samples) for seq_len={self.seq_len} and pred_len={self.pred_len}")

        for i in range(max_start_index + 1):
            X.append(features[i:(i + self.seq_len), :])
            # Target is the value(s) *after* the input sequence
            y.append(target[(i + self.seq_len):(i + self.seq_len + self.pred_len)])

        if not X:
             # This case should be caught by the length check above, but as a safeguard:
            raise ValueError("Could not create any sequences. Check dataset length, seq_len, and pred_len.")


        # Ensure target has the correct shape: (num_sequences, pred_len, 1)
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1) # Add feature dim
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def context_length(self):
        return self.seq_len

# New Generator Class implementing DataGen
class BatteryDatasetGenerator(DataGen):
    config_class = BatteryDatasetConfig

    def __init__(self, cfg: BatteryDatasetConfig):
        self.cfg = cfg

        # Load the full training dataset
        full_train_dataset = BatteryDataset(
            file_path=cfg.train_file_path,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            target_column=cfg.target_column,
            feature_columns=cfg.feature_columns,
        )

        # Perform train/validation split
        train_size = int((1.0 - cfg.val_split_percent) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        # Ensure sizes are non-negative
        if train_size < 0 or val_size < 0:
             raise ValueError(f"Calculated invalid train/validation split sizes: train={train_size}, val={val_size}")

        self._train_ds, self._val_ds = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Store input/output dimensions from the dataset
        self.input_dim = full_train_dataset.input_dim
        self.output_dim = full_train_dataset.output_dim
        self._context_length = full_train_dataset.context_length # Store context length

        # Define metrics appropriate for regression
        self._metrics = torchmetrics.MetricCollection([
            MeanSquaredError(),
            MeanAbsoluteError()
        ])
        print(f"BatteryDatasetGenerator initialized: Train size={train_size}, Val size={val_size}, Input dim={self.input_dim}, Output dim={self.output_dim}")


    @property
    def train_split(self) -> Dataset:
        return self._train_ds

    @property
    def validation_split(self) -> Mapping[str, Dataset]:
        # Adhere to the expected dictionary format
        return {"val": self._val_ds}

    @property
    def train_metrics(self) -> torchmetrics.MetricCollection:
        # Return a clone to ensure independent state per use (e.g., train vs val)
        return self._metrics.clone()

    @property
    def validation_metrics(self) -> torchmetrics.MetricCollection:
        # Return a clone
        return self._metrics.clone()

    # Add context_length property needed by main script? Check main.py usage.
    # main.py accesses context_length from the model config (cfg.model.context_length)
    # However, the original FormLangDatasetGenerator had it. Let's add it for consistency.
    @property
    def context_length(self) -> int:
        # Get it from one of the underlying datasets (or stored config)
        # Accessing Subset's dataset attribute
         if isinstance(self._train_ds, Subset):
             return self._train_ds.dataset.context_length
         else:
              # Should not happen with random_split unless split is 0/100%
             return self._train_ds.context_length
