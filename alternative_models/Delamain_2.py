import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase

class Delamain_2(DelamainBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 21 * 21 * 2, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('pre permute:', x.size())
        x = x.transpose(2,3).permute(1, 0, 4, 2, 3)
        # print('post permute:', x.size())
        # Permute the dimensions to have channels first (batch, channels, height, width)
        current = x[0]
        current = self.pool(F.relu(self.conv1(current)))
        current = F.relu(self.conv2(current))
        current = self.pool(F.relu(self.conv3(current)))

        past = x[1]
        past = self.pool(F.relu(self.conv1(past)))
        past = F.relu(self.conv2(past))
        past = self.pool(F.relu(self.conv3(past)))

        out = [current, past]
        out = torch.cat(out, 1) # concat branches

        out = torch.flatten(out, 1) # flatten all dimensions except batch
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def is_prev_frame_needed(self) -> bool:
        return True