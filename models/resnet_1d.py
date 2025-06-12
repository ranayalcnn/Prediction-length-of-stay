import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.in_channels = in_channels

        self.block1 = ResidualBlock(in_channels, 64, dropout=0.2)
        self.block2 = ResidualBlock(64, 128, dropout=0.3)
        self.block3 = ResidualBlock(128, 256, dropout=0.4)
        self.block4 = ResidualBlock(256, 384, dropout=0.5)

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        current_channels = x.shape[1]
        if current_channels < self.in_channels:
            pad_channels = self.in_channels - current_channels
            x = F.pad(x, (0, 0, 0, pad_channels))
        elif current_channels > self.in_channels:
            x = x[:, :self.in_channels, :]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        avg_pool = self.pool_avg(x).squeeze(-1)
        max_pool = self.pool_max(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(x)
