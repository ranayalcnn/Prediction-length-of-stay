import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCNN(nn.Module):
    def __init__(self, in_channels=8):
        super(DeepCNN, self).__init__()
        self.in_channels = in_channels

        # Daha dengeli filtre sayıları ve daha kısa mimari
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Global pooling (ortalama ve maksimum)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        # Daha sade regressor
        self.regressor = nn.Sequential(
            nn.Linear(256, 64),  # 128*2 (avg + max)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        # Kanal uyumsuzluğu durumunda padding
        current_channels = x.shape[1]
        if current_channels < self.in_channels:
            pad = self.in_channels - current_channels
            x = F.pad(x, (0, 0, 0, pad))
        elif current_channels > self.in_channels:
            x = x[:, :self.in_channels, :]

        x = self.features(x)
        avg_pool = self.pool_avg(x).squeeze(-1)
        max_pool = self.pool_max(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.regressor(x)
