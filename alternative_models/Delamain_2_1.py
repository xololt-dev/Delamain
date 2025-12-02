import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase

class Delamain_2_1(DelamainBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 21 * 21, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)

        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.batch_norm2(self.conv3(x))))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def is_prev_frame_needed(self) -> bool:
        return True