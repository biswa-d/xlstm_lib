# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Andreas Auer (adapted for regression)
from dataclasses import dataclass
from typing import Sequence # Keep if using WeightDecayOptimGroupMixin

import torch
from torch import nn

from .components.init import kaiming_uniform_init_, xavier_uniform_init_ # Import suitable initializers
from .utils import WeightDecayOptimGroupMixin # Optional: If fine-tuning weight decay
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMRegressionModelConfig(xLSTMBlockStackConfig):
    """Configuration specific to the xLSTM Regression Model."""
    input_dim: int = -1  # Number of input features (e.g., 3 for SOC, Temp, Current)
    output_dim: int = -1 # Number of output values to predict (e.g., 1 for Voltage)
    # Optional: Add flags for specific initializations, layer norms, etc.
    # weight_decay_on_proj: bool = False # Example if using WeightDecayOptimGroupMixin


# Optional: Inherit from WeightDecayOptimGroupMixin if you want separate weight decay 
# settings for the projection layers, similar to the embedding in xLSTMLMModel.
class xLSTMRegressionModel(nn.Module):
    """ A wrapper around xLSTMBlockStack for time-series regression tasks. """
    config_class = xLSTMRegressionModelConfig

    def __init__(self, config: xLSTMRegressionModelConfig, **kwargs):
        super().__init__()
        if config.input_dim <= 0:
            raise ValueError("'input_dim' must be positive.")
        if config.output_dim <= 0:
            raise ValueError("'output_dim' must be positive.")
        self.config = config

        # 1. Input projection layer: Maps input features to embedding dimension
        self.input_proj = nn.Linear(config.input_dim, config.embedding_dim)
        # Optional: Add activation or LayerNorm here if needed
        # self.input_act = nn.ReLU() # Example
        # self.input_norm = nn.LayerNorm(config.embedding_dim) # Example

        # 2. Core xLSTM Stack
        # The stack expects input shape (batch, seq_len, embedding_dim)
        self.xlstm_block_stack = xLSTMBlockStack(config=config)

        # 3. Output projection layer (Regression Head)
        # Maps final hidden states to the desired output dimension
        self.output_proj = nn.Linear(config.embedding_dim, config.output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights of the model."""
        # print(f"Initializing {self.__class__.__name__}...")
        # Initialize input projection layer (example using Kaiming Uniform)
        # kaiming_uniform_init_(self.input_proj.weight, nonlinearity='linear') # Use 'relu' if using ReLU activation
        # if self.input_proj.bias is not None:
        #     nn.init.zeros_(self.input_proj.bias)
        # Simple initialization for now:
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
             nn.init.zeros_(self.input_proj.bias)
        
        # Reset the stack's parameters (it handles its internal initialization)
        self.xlstm_block_stack.reset_parameters()

        # Initialize output projection layer (example using Xavier Uniform)
        xavier_uniform_init_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        # print("Initialization complete.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training and evaluation.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.config.input_dim, \
               f"Input dimension mismatch: got {input_dim}, expected {self.config.input_dim}"

        # 1. Project input features to embedding dimension
        # Input x: (batch, seq_len, input_dim)
        embedded_x = self.input_proj(x)
        # embedded_x: (batch, seq_len, embedding_dim)
        
        # Optional: Apply activation/normalization
        # embedded_x = self.input_act(embedded_x) 
        # embedded_x = self.input_norm(embedded_x)

        # 2. Process sequence through xLSTM stack
        # Input embedded_x: (batch, seq_len, embedding_dim)
        hidden_states = self.xlstm_block_stack(embedded_x)
        # hidden_states: (batch, seq_len, embedding_dim)

        # 3. Project final hidden states to output dimension
        # Input hidden_states: (batch, seq_len, embedding_dim)
        predictions = self.output_proj(hidden_states)
        # predictions: (batch, seq_len, output_dim)

        # The shape of the output matches the input sequence length.
        # The loss function (e.g., MSELoss) will compare this against the target sequence.
        # If your targets (`y` from BatteryDataset) have shape (batch, pred_len, output_dim),
        # ensure the loss calculation correctly compares `predictions[:, -pred_len:, :]` with `y`.
        # Or, adjust the target creation in BatteryDataset if needed.
        # For now, returning the full sequence prediction is standard.
        return predictions

    # Optional: Implement step method if needed for autoregressive inference
    # def step(self, x_step: torch.Tensor, state: dict = None, **kwargs) -> tuple[torch.Tensor, dict]:
    #     """ Process a single time step. """
    #     # Input x_step: (batch, 1, input_dim)
    #     embedded_x_step = self.input_proj(x_step)
    #     hidden_state_step, state = self.xlstm_block_stack.step(embedded_x_step, state=state, **kwargs)
    #     prediction_step = self.output_proj(hidden_state_step)
    #     # prediction_step: (batch, 1, output_dim)
    #     return prediction_step, state

    # Optional: Implement _create_weight_decay_optim_groups if using WeightDecayOptimGroupMixin
    # def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
    #     weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
    #     # Add logic to handle weight decay for self.input_proj and self.output_proj based on config flags
    #     return weight_decay, no_weight_decay
