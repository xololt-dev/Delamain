import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase

class Delamain_2_1(DelamainBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(16 * 21 * 21, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Permute the dimensions to have channels first (batch, channels, height, width)
    #     x = x.permute(0, 3, 1, 2)

    #     x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(F.relu(self.batch_norm2(self.conv3(x))))

    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute the dimensions to have channels first (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)

        current = x[:, :6, :, :]
        current = self.pool(F.relu(self.batch_norm1(self.conv1(current))))
        current = F.relu(self.conv2(current))
        current = self.pool(F.relu(self.batch_norm2(self.conv3(current))))

        past = x[:, 6:, :, :]
        past = self.pool(F.relu(self.batch_norm1(self.conv1(past))))
        past = F.relu(self.conv2(past))
        past = self.pool(F.relu(self.batch_norm2(self.conv3(past))))

        out = [current, past]
        out = torch.cat(out, 1) # concat branches

        out = torch.flatten(out, 1) # flatten all dimensions except batch
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def is_prev_frame_needed(self) -> bool:
        return True