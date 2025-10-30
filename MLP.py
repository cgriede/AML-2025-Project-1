import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

# Define a simple MLP module
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units=128, num_layers=2, dropout=0.2):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_units), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(1, num_layers):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_units, 1))  # Regression output
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)