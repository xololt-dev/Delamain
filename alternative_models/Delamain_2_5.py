import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase


class Delamain_2_5(DelamainBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=4,
            stride=2,
            dtype=torch.float32,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=4, dtype=torch.float32
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=4,
            stride=2,
            dtype=torch.float32,
        )
        self.fc1 = nn.Linear(32 * 21 * 21, 256, dtype=torch.float32)
        self.fc2 = nn.Linear(256, 5, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute the dimensions to have channels first (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0

        # 47x47
        x = F.relu(self.conv1(x))
        # 44x44
        x = F.relu(self.conv2(x))
        # 21x21
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def is_prev_frame_needed(self) -> bool:
        return True

    def prev_frames_needed(self) -> int:
        return 4
