from torch import nn
import torch.nn.functional as F
from trainer.constants import INPUT_DIM, HIDDEN_DIM, NUM_CLASSES

# Simple MLP model for MNIST classification
# 2-layer neural network with ReLU activation
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        # input layer to hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # hidden layer to output layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # first layer with ReLU
        x = F.relu(self.fc1(x))
        # output layer (no activation, will use CrossEntropyLoss)
        x = self.fc2(x)
        return x