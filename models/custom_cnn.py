import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, in_channels=8):
        super(CustomCNN, self).__init__()
        self.in_channels = in_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
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
        x = self.feature_extractor(x)
        avg_pool = self.pool_avg(x).squeeze(-1)
        max_pool = self.pool_max(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.regressor(x)
