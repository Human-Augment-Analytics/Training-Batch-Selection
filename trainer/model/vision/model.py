from torch import nn
import torch.nn.functional as F
from trainer.constants import INPUT_DIM, HIDDEN_DIM, NUM_CLASSES

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Basic CNN added by LMT to test higher dimensional datasets - feel free to swap out
class SimpleCNN(nn.Module):
    """Works for MNIST (1×28×28) and CIFAR10 (3×32×32)."""
    # this model expects images, not flat vectors
    expects_flat = False

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        print (f'[model.py]: building a CNN with in_channels={in_channels} and num_classes={num_classes}')
        self.in_channels=in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # /2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # /4 total
        )
        self.gap = nn.AdaptiveAvgPool2d(1)       # -> (B,128,1,1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        assert x.dim()==4 and x.shape[1]==self.in_channels, f"Got {tuple(x.shape)}"
        x = self.features(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (B,128)
        return self.classifier(x)
