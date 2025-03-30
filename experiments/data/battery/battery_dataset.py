import pandas as pd
import torch
import torchmetrics
from torch.utils.data import Dataset
from dataclasses import dataclass
from ..utils import DataGen
from ...metrics import SequenceAccuracy

# Configuration class for Battery Dataset
@dataclass
class BatteryDatasetConfig:
    train_file_path: str
    test_file_path: str
    seq_len: int
    pred_len: int
    target_column: str
    shift: int = 1
    enable_mask: bool = False
    additional_prefix_tokens: int = 0
    additional_suffix_tokens: int = 0
    additional_premask_tokens: int = 0

# BatteryDataset class definition
class BatteryDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        pred_len: int,
        target_column: str,
        shift: int = 1,
        enable_mask: bool = False,
        additional_prefix_tokens: int = 0,
        additional_suffix_tokens: int = 0,
        additional_premask_tokens: int = 0
    ):
        self.data = pd.read_csv(file_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_column = target_column
        self.shift = shift
        self.enable_mask = enable_mask
        self.additional_prefix_tokens = additional_prefix_tokens
        self.additional_suffix_tokens = additional_suffix_tokens
        self.additional_premask_tokens = additional_premask_tokens

        # Extract features (Current, Temp, SOC) and target (Voltage)
        self.features = self.data[['Current', 'Temp', 'SOC']].values
        self.target = self.data[self.target_column].values

        self.X, self.y = self.create_sequences(self.features, self.target)

    def create_sequences(self, features, target):
        X, y = [], []
        for i in range(len(features) - self.seq_len - self.pred_len + 1):
            X.append(features[i:(i + self.seq_len), :])
            y.append(target[i + self.seq_len:(i + self.seq_len + self.pred_len)])

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, self.pred_len, 1)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def vocab_size(self):
        return self.features.shape[1]

    @property
    def context_length(self):
        return self.seq_len

# Function to load battery datasets for training and testing
def load_battery_datasets(config: BatteryDatasetConfig):
    # Load the training dataset
    train_dataset = BatteryDataset(
        file_path=config.train_file_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        target_column=config.target_column,
        shift=config.shift,
        enable_mask=config.enable_mask,
        additional_prefix_tokens=config.additional_prefix_tokens,
        additional_suffix_tokens=config.additional_suffix_tokens,
        additional_premask_tokens=config.additional_premask_tokens
    )

    # Load the testing dataset
    test_dataset = BatteryDataset(
        file_path=config.test_file_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        target_column=config.target_column,
        shift=config.shift,
        enable_mask=config.enable_mask,
        additional_prefix_tokens=config.additional_prefix_tokens,
        additional_suffix_tokens=config.additional_suffix_tokens,
        additional_premask_tokens=config.additional_premask_tokens
    )

    # Split the training dataset into train and validation sets (e.g., 85/15 split)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    return train_dataset, val_dataset, test_dataset

# Metrics for training, validation, and testing
def get_metrics(vocab_size):
    return torchmetrics.MetricCollection(
        SequenceAccuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-1
        )
    )
