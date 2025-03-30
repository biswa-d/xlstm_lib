# Copyright (c) NXAI GmbH and its affiliates 2024
# Andreas Auer, Maximilian Beck
from abc import abstractmethod
from typing import Tuple, Any, Optional, Mapping

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torchmetrics
from torch.utils.data import TensorDataset, Dataset, Subset, random_split

from torchmetrics import MeanSquaredError, MeanAbsoluteError


class DataGen:

    @property
    @abstractmethod
    def train_split(self) -> torch.utils.data.Dataset:
        pass

    @property
    @abstractmethod
    def validation_split(self) -> Mapping[str, torch.utils.data.Dataset]:
        pass

    @property
    @abstractmethod
    def train_metrics(self) -> torchmetrics.MetricCollection:
        pass

    @property
    @abstractmethod
    def validation_metrics(self) -> torchmetrics.MetricCollection:
        pass

class SequenceTensorDataset(TensorDataset):

    def __init__(self, tensors: Tuple[Any, Any], vocab_size: int, context_length: int) -> None:
        super().__init__(*tensors)
        self._vocab_size = vocab_size
        self._context_length = context_length

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab_size

    @property
    def context_length(self) -> int:
        return self._context_length


class CacheMixin:

    @staticmethod
    def check_exist(config, directory: Path, check_existing: bool):
        if directory is not None and os.path.exists(str(directory / "config.json")):
            # check if
            if check_existing:
                ds_hash = CacheMixin.dataset_hash(directory)
                with open(str(directory / "config.json")) as fp:
                    read_conf = json.load(fp)
                conf_dict = asdict(config)
                conf_dict["hash"] = ds_hash
                # compare except for data dir
                del read_conf["data_dir"]
                del conf_dict["data_dir"]
                assert read_conf == conf_dict, (
                    f"Non-matching configuration: " f"Read: {read_conf} - Current: {conf_dict}"
                )
            return True
        else:
            return False

    @staticmethod
    def post_generate(config, directory):
        ds_hash = CacheMixin.dataset_hash(directory)
        conf_dict = asdict(config)
        conf_dict["hash"] = ds_hash
        with open(str(directory / "config.json"), "w") as fp:
            json.dump(conf_dict, fp)

    @staticmethod
    def dataset_hash(subdir):
        return calc_joint_md5sum(subdir, exclude=["config.json"])


def calc_joint_md5sum(dir_path, exclude=[]):
    md5 = hashlib.md5()
    file_names = sorted(os.listdir(dir_path))
    for file_name in file_names:
        if file_name in exclude:
            continue
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                file_data = f.read()
                md5.update(file_data)
    return md5.hexdigest()


# Configuration class for Battery Dataset
@dataclass
class BatteryDatasetConfig:
    train_file_path: str
    test_file_path: str # Keep test path here, but loading might happen differently in main.py
    seq_len: int
    pred_len: int
    target_column: str
    # Define input columns explicitly
    feature_columns: Tuple[str, ...] = ('Current', 'Temp', 'SOC')
    # Validation split percentage
    val_split_percent: float = 0.15


# BatteryDataset class definition - Remains largely the same, inherits from Dataset
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
        # Total length needed = seq_len (input) + pred_len (output)
        total_len_needed = self.seq_len + self.pred_len
        if len(features) < total_len_needed:
             raise ValueError(f"Dataset too short ({len(features)} samples) for seq_len={self.seq_len} and pred_len={self.pred_len}. Need at least {total_len_needed} samples.")

        # Iterate up to the last possible starting point for a full sequence + prediction
        for i in range(len(features) - total_len_needed + 1):
            X.append(features[i:(i + self.seq_len), :])
            # Target is the value(s) *after* the input sequence ends
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

    # context_length property might be needed if accessed directly from dataset instance
    @property
    def context_length(self):
        return self.seq_len

    @property
    def num_features(self):
        return self.input_dim


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

        # Ensure sizes are non-negative and valid for splitting
        if train_size <= 0 or val_size <= 0:
             raise ValueError(f"Calculated invalid train/validation split sizes based on percentage {cfg.val_split_percent}: train={train_size}, val={val_size}. Check dataset size and val_split_percent.")

        self._train_ds, self._val_ds = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Store input/output dimensions and context length from the dataset
        self.input_dim = full_train_dataset.input_dim
        self.output_dim = full_train_dataset.output_dim
        self._context_length = full_train_dataset.context_length # Store context length

        # Define metrics appropriate for regression
        self._metrics = torchmetrics.MetricCollection([
            MeanSquaredError(),
            MeanAbsoluteError()
        ])
        print(f"BatteryDatasetGenerator initialized: Train size={train_size}, Val size={val_size}, Input dim={self.input_dim}, Output dim={self.output_dim}, Context len={self._context_length}")


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

    @property
    def context_length(self) -> int:
         # Return the stored context length
         return self._context_length

    @property
    def num_features(self) -> int:
        # Provide the input feature dimension
        return self.input_dim


# Removed load_battery_datasets function
# Removed get_metrics function

