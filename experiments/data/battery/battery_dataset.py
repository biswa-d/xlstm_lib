import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class BatteryDatasetConfig:
    file_path: str
    seq_len: int
    pred_len: int
    target_column: str

class BatteryDataset(Dataset):
    config_class = BatteryDatasetConfig  # Add this line

    def __init__(self, config: BatteryDatasetConfig):
        self.data = pd.read_csv(config.file_path)
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.target_column = config.target_column

        # Extract features (Current, Temp, SOC) and target (Voltage)
        self.features = self.data[['Current', 'Temp', 'SOC']].values  # Removed Voltage
        self.target = self.data[self.target_column].values  # Assuming you want to predict 'Voltage'

        self.X, self.y = self.create_sequences(self.features, self.target)

    def create_sequences(self, features, target):
        X, y = [], []
        for i in range(len(features) - self.seq_len - self.pred_len + 1):
            X.append(features[i:(i + self.seq_len), :])  # Take sequence of length `seq_len`
            y.append(target[i + self.seq_len:(i + self.seq_len + self.pred_len)])  # Predict next `pred_len` step(s)

        # Convert to PyTorch tensors with appropriate dimensions
        return (
            torch.tensor(X, dtype=torch.float32),  # Shape: (num_samples, seq_len, num_features)
            torch.tensor(y, dtype=torch.float32).view(-1, self.pred_len, 1)  # Shape: (num_samples, pred_len, 1)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
