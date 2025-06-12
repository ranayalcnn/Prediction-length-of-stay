import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.3):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.se = SEBlock(96)

    def forward(self, x):
        out1 = self.branch1(x)
        out3 = self.branch3(x)
        out5 = self.branch5(x)
        out = torch.cat([out1, out3, out5], dim=1)
        out = self.se(out)
        out = self.dropout(out)
        return out

class Inception1D(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.in_channels = in_channels
        self.inception1 = InceptionBlock(in_channels, dropout_rate=0.3)
        self.inception2 = InceptionBlock(96, dropout_rate=0.4)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def pad_channels(self, x):
        current_channels = x.shape[1]
        if current_channels == self.in_channels:
            return x
        elif current_channels < self.in_channels:
            pad = self.in_channels - current_channels
            noise = torch.randn(x.size(0), pad, x.size(2), device=x.device) * 0.1
            x = torch.cat([x, noise], dim=1)
        else:
            x = x[:, :self.in_channels, :]
        return x

    def forward(self, x):
        x = self.pad_channels(x)
        x = self.inception1(x)
        x = self.inception2(x)
        avg_pool = self.pool_avg(x).squeeze(-1)
        max_pool = self.pool_max(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(x)
