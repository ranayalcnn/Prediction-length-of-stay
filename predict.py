import torch
import numpy as np

class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

def predict(npz_path):
    data = np.load(npz_path)
    X = torch.tensor(data["X"], dtype=torch.float32).permute(0, 2, 1)
    in_channels = X.shape[1]

    model = SimpleCNN(in_channels=in_channels)
    model_path = f"model/los_model_{in_channels}ch.pt"

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(X).squeeze().numpy()
        print(f"✅ {len(preds)} segment tahmin edildi.")
        print(f"📊 Ortalama tahmini LOS: {np.mean(preds):.2f} gün")

if __name__ == "__main__":
    predict("data/processed_3ch/3743522_0006.npz")

