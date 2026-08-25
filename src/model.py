import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Globally averages each channel, applies an FC bottleneck,
    and rescales the original channels.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        s = x.mean(dim=[2, 3])
        s = self.fc(s).unsqueeze(-1).unsqueeze(-1)

        return x * s


class ResBlock(nn.Module):
    """
    Two-layer residual convolutional block with
    Squeeze-and-Excitation attention.
    """
    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SEBlock(channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)

        return F.relu(out + residual)


class DownsampleBlock(nn.Module):
    """
    Stride-2 convolutional block that halves spatial
    dimensions and increases the number of channels.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),

            nn.Conv2d(
                out_ch, out_ch,
                3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                1, stride=2, bias=False
            ),
            nn.BatchNorm2d(out_ch),
        )

        self.se = SEBlock(out_ch)

    def forward(self, x):
        return F.relu(
            self.se(self.conv(x)) + self.shortcut(x)
        )


class AudioCNNGRU(nn.Module):
    """
    Residual CNN + Bidirectional GRU architecture.

    Input:
        (B, 3, 128, T)

    Output:
        (B, 10)
    """
    def __init__(self, num_classes=10, gru_hidden=256):
        super().__init__()

        # CNN stem: 3 -> 32 channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 32,
                3, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Residual stages
        self.stage1 = nn.Sequential(
            DownsampleBlock(32, 64),
            ResBlock(64)
        )

        self.stage2 = nn.Sequential(
            DownsampleBlock(64, 128),
            ResBlock(128)
        )

        self.stage3 = nn.Sequential(
            DownsampleBlock(128, 256),
            ResBlock(256)
        )

        # Preserve temporal dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((4, None))

        gru_in = 256 * 4

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, 3, 128, T)

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # (B, 256, 4, T/8)
        x = self.freq_pool(x)

        B, C, F, T = x.shape

        # (B, T, C, F)
        x = x.permute(0, 3, 1, 2)

        # (B, T, 1024)
        x = x.reshape(B, T, C * F)

        # (B, T, 512)
        x, _ = self.gru(x)

        # Temporal mean pooling
        x = x.mean(dim=1)

        # 10 digit logits
        return self.head(x)
