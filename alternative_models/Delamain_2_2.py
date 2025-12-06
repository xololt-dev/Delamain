import torch
import torch.nn as nn
import torch.nn.functional as F

from .DelamainBase import DelamainBase

class Delamain_2_2(DelamainBase):
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
    #     self.pool = nn.MaxPool2d(2)
    #     self.batch_norm1 = nn.BatchNorm2d(24)
    #     self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3)
    #     self.batch_norm2 = nn.BatchNorm2d(16)
    #     self.conv3 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3)
    #     self.fc1 = nn.Linear(16 * 21 * 21, 100)
    #     self.fc2 = nn.Linear(100, 60)
    #     self.fc3 = nn.Linear(60, 5)

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

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 21 * 21 * 4, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('pre permute:', x.size())
        x = x.permute(0, 3, 1, 2)
        # print('post permute:', x.size())
        # Permute the dimensions to have channels first (batch, channels, height, width)
        current = x[:,9:12,:,:]
        current = self.pool(F.relu(self.batch_norm1(self.conv1(current))))
        current = F.relu(self.conv2(current))
        current = self.pool(F.relu(self.batch_norm2(self.conv3(current))))

        past_1 = x[:,:3,:,:]
        past_1 = self.pool(F.relu(self.batch_norm1(self.conv1(past_1))))
        past_1 = F.relu(self.conv2(past_1))
        past_1 = self.pool(F.relu(self.batch_norm2(self.conv3(past_1))))
        past_2 = x[:,3:6,:,:]
        past_2 = self.pool(F.relu(self.batch_norm1(self.conv1(past_2))))
        past_2 = F.relu(self.conv2(past_2))
        past_2 = self.pool(F.relu(self.batch_norm2(self.conv3(past_2))))
        past_3 = x[:,6:9,:,:]
        past_3 = self.pool(F.relu(self.batch_norm1(self.conv1(past_3))))
        past_3 = F.relu(self.conv2(past_3))
        past_3 = self.pool(F.relu(self.batch_norm2(self.conv3(past_3))))

        out = [current, past_1, past_2, past_3]
        out = torch.cat(out, 1) # concat branches

        out = torch.flatten(out, 1) # flatten all dimensions except batch
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def is_prev_frame_needed(self) -> bool:
        return True