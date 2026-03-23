import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase


class Delamain_2_1(DelamainBase):
    def __init__(self, input_size=96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=3)

        with torch.no_grad():
            dummy = torch.zeros(1, 6, input_size, input_size)
            dummy = self._forward_branch(dummy)
            self._fc_input_size = dummy.numel() * 2  # two branches concatenated

        self.fc1 = nn.Linear(self._fc_input_size, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    def _forward_branch(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.batch_norm2(self.conv3(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute the dimensions to have channels first (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0

        current = x[:, :6, :, :]
        current = self._forward_branch(current)

        past = x[:, 6:, :, :]
        past = self._forward_branch(past)

        out = [current, past]
        out = torch.cat(out, 1)  # concat branches

        out = torch.flatten(out, 1)  # flatten all dimensions except batch
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def is_prev_frame_needed(self) -> bool:
        return True
