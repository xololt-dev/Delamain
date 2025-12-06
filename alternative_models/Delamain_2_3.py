import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase

class Delamain_2_3(DelamainBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2)
        # self.batch_norm1 = nn.BatchNorm2d(24, dtype=torch.float32)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=4, padding="same", dtype=torch.float32)
        # self.batch_norm2 = nn.BatchNorm2d(16, dtype=torch.float32)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=4, dtype=torch.float32)
        self.fc1 = nn.Linear(16 * 10 * 10, 160, dtype=torch.float32)
        self.fc2 = nn.Linear(160, 32, dtype=torch.float32)
        self.fc3 = nn.Linear(32, 5, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.half()
        # Permute the dimensions to have channels first (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)

        # 23x23
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        # 10x10 ?
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.batch_norm2(self.conv3(x))))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def is_prev_frame_needed(self) -> bool:
        return True