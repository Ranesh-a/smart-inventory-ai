"""
 Multivariate LSTM Model for Smart Inventory Demand Forecasting.
 PyTorch implementation with GPU support for NVIDIA RTX 2050.
 Input features: sales, price, weekday, month, is_weekend, is_event_day (N=6).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMModel(nn.Module):
    """
    LSTM-based neural network for demand forecasting.
    
    Architecture:
        - LSTM layer (hidden_size=50, num_layers=1)
        - Fully Connected layer for output
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 50,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features (default: 6 for multivariate).
            hidden_size: Number of hidden units in LSTM (default: 50).
            num_layers: Number of LSTM layers (default: 1).
            output_size: Number of output values (default: 1 for single-day prediction).
            dropout: Dropout rate between LSTM layers (default: 0.0).
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden states on the correct device (GPU/CPU).
        
        Args:
            batch_size: Current batch size.
            device: Device to place tensors on (cuda or cpu).
            
        Returns:
            Tuple of (h0, c0) hidden state tensors.
        """
        # Hidden state: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            hidden: Optional tuple of (h0, c0) hidden states.
            
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)
        device = x.device  # Get device from input tensor
        
        # Initialize hidden states on the correct device if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, device)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Take the output from the last time step
        # last_out shape: (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        # output shape: (batch_size, output_size)
        output = self.fc(last_out)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction (inference mode).
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Prediction tensor of shape (batch_size, output_size).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA GPU if available, else CPU).
    
    Returns:
        torch.device object for GPU or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    return device


# Quick test when running directly
if __name__ == "__main__":
    # Get device
    device = get_device()
    
    # Create model and move to device
    model = LSTMModel(input_size=6, hidden_size=50, num_layers=1, output_size=1)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 64
    seq_length = 30
    n_features = 6
    test_input = torch.randn(batch_size, seq_length, n_features, device=device)
    
    output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
