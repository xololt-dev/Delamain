import torch
import torch.nn as nn
import torch.nn.functional as F
from .DelamainBase import DelamainBase


# source: https://github.com/d-li14/mobilenetv4.pytorch
def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dtype=torch.float32
    ):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                (kernel_size - 1) // 2,
                bias=False,
                dtype=dtype,
            ),
            nn.BatchNorm2d(out_channels, dtype=dtype),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        start_dw_kernel_size,
        middle_dw_kernel_size,
        stride,
        middle_dw_downsample: bool = True,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        dtype=torch.float32,
    ):
        super(UniversalInvertedBottleneck, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
            self.start_dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                start_dw_kernel_size,
                stride if not middle_dw_downsample else 1,
                (start_dw_kernel_size - 1) // 2,
                groups=in_channels,
                bias=False,
                dtype=dtype,
            )
            self.start_dw_norm = nn.BatchNorm2d(in_channels, dtype=dtype)

        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(
            in_channels,
            expand_channels,
            1,
            1,
            bias=False,
            dtype=dtype,
        )
        self.expand_norm = nn.BatchNorm2d(expand_channels, dtype=dtype)
        self.expand_act = nn.ReLU(
            inplace=True,
        )

        if middle_dw_kernel_size:
            self.middle_dw_conv = nn.Conv2d(
                expand_channels,
                expand_channels,
                middle_dw_kernel_size,
                stride if middle_dw_downsample else 1,
                (middle_dw_kernel_size - 1) // 2,
                groups=expand_channels,
                bias=False,
                dtype=dtype,
            )
            self.middle_dw_norm = nn.BatchNorm2d(expand_channels, dtype=dtype)
            # self.middle_dw_norm = nn.GroupNorm(num_groups=8, num_channels=expand_channels, dtype=dtype)
            self.middle_dw_act = nn.ReLU(
                inplace=True,
            )

        self.proj_conv = nn.Conv2d(
            expand_channels,
            out_channels,
            1,
            1,
            bias=False,
            dtype=dtype,
        )
        self.proj_norm = nn.BatchNorm2d(out_channels, dtype=dtype)

        if use_layer_scale:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True
            )

        self.use_layer_scale = use_layer_scale
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.identity else x


class MobileNetV4(DelamainBase):
    def __init__(self, in_channels=12, input_size=96):
        super().__init__()

        # 96 / 84

        # 48 / 42
        self.conv1 = ConvBN(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            dtype=torch.float32,
        )
        # 24 / 21
        self.conv2 = ConvBN(
            in_channels=32,
            out_channels=96,
            kernel_size=3,
            stride=2,
            dtype=torch.float32,
        )
        self.conv3 = ConvBN(
            in_channels=96,
            out_channels=64,
            kernel_size=1,
            stride=1,
            dtype=torch.float32,
        )
        # 12 / 10
        self.universal1 = UniversalInvertedBottleneck(
            in_channels=64,
            out_channels=96,
            expand_ratio=3.0,
            start_dw_kernel_size=5,
            middle_dw_kernel_size=5,
            stride=2,
            dtype=torch.float32,
        )
        self.universal2 = UniversalInvertedBottleneck(
            in_channels=96,
            out_channels=96,
            expand_ratio=2.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=3,
            stride=1,
            dtype=torch.float32,
        )
        self.universal3 = UniversalInvertedBottleneck(
            in_channels=96,
            out_channels=96,
            expand_ratio=2.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=3,
            stride=1,
            dtype=torch.float32,
        )
        self.universal4 = UniversalInvertedBottleneck(
            in_channels=96,
            out_channels=96,
            expand_ratio=2.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=3,
            stride=1,
            dtype=torch.float32,
        )
        self.universal5 = UniversalInvertedBottleneck(
            in_channels=96,
            out_channels=96,
            expand_ratio=2.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=3,
            stride=1,
            dtype=torch.float32,
        )
        self.universal6 = UniversalInvertedBottleneck(
            in_channels=96,
            out_channels=96,
            expand_ratio=4.0,
            start_dw_kernel_size=3,
            middle_dw_kernel_size=0,
            stride=1,
            dtype=torch.float32,
        )
        # 6 / 6
        self.universal7 = UniversalInvertedBottleneck(
            in_channels=96,
            out_channels=128,
            expand_ratio=6.0,
            start_dw_kernel_size=3,
            middle_dw_kernel_size=3,
            stride=2,
            dtype=torch.float32,
        )
        self.universal8 = UniversalInvertedBottleneck(
            in_channels=128,
            out_channels=128,
            expand_ratio=4.0,
            start_dw_kernel_size=5,
            middle_dw_kernel_size=5,
            stride=1,
            dtype=torch.float32,
        )
        self.universal9 = UniversalInvertedBottleneck(
            in_channels=128,
            out_channels=128,
            expand_ratio=4.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=5,
            stride=1,
            dtype=torch.float32,
        )
        self.universal10 = UniversalInvertedBottleneck(
            in_channels=128,
            out_channels=128,
            expand_ratio=3.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=5,
            stride=1,
            dtype=torch.float32,
        )
        self.universal11 = UniversalInvertedBottleneck(
            in_channels=128,
            out_channels=128,
            expand_ratio=4.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=3,
            stride=1,
            dtype=torch.float32,
        )
        self.universal12 = UniversalInvertedBottleneck(
            in_channels=128,
            out_channels=128,
            expand_ratio=4.0,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=3,
            stride=1,
            dtype=torch.float32,
        )
        self.conv4 = ConvBN(
            in_channels=128,
            out_channels=960,
            kernel_size=1,
            stride=1,
            dtype=torch.float32,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_channels = 1280
        # with torch.no_grad():
        #     dummy = torch.zeros(
        #         1, in_channels, input_size, input_size, dtype=torch.float32
        #     )
        #     dummy = self._forward_conv(dummy)
        #     self._fc_input_size = dummy.numel()
        # self.conv = ConvBN(
        #     self._fc_input_size,
        #     hidden_channels,
        #     1,
        #     dtype=torch.float32,
        # )
        self.conv = ConvBN(960, hidden_channels, 1)
        # self.conv = nn.Conv2d(960, hidden_channels, 1, bias=True)
        self.classifier = nn.Linear(
            hidden_channels,
            5,
            dtype=torch.float32,
        )

    def _forward_conv(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = self.universal1(x)
        x = self.universal2(x)
        x = self.universal3(x)
        x = self.universal4(x)
        x = self.universal5(x)
        x = self.universal6(x)

        x = self.universal7(x)
        x = self.universal8(x)
        x = self.universal9(x)
        x = self.universal10(x)
        x = self.universal11(x)
        x = self.universal12(x)
        x = self.conv4(x)
        x = self.avgpool(x)

        return x

    def forward(self, x):
        # print(f" input is: {x.size()}", end=" ")

        x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0

        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = self.universal1(x)
        x = self.universal2(x)
        x = self.universal3(x)
        x = self.universal4(x)
        x = self.universal5(x)
        x = self.universal6(x)

        x = self.universal7(x)
        x = self.universal8(x)
        x = self.universal9(x)
        x = self.universal10(x)
        x = self.universal11(x)
        x = self.universal12(x)
        x = self.conv4(x)

        # x = self.features(x)
        # print("before avgpool", x.size())
        x = self.avgpool(x)
        # print(x.size())
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def action_mode_switch(self) -> bool:
        return True
