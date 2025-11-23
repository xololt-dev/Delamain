import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase

class Delamain(DelamainBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 21 * 21, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    def forward(self, x):
        # Permute the dimensions to have channels first (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x