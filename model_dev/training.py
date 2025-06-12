import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class SegmentDataset(Dataset):
    def __init__(self, data_dir, id_list_path):
        with open(id_list_path, "r") as f:
            allowed_ids = {line.strip() for line in f}

        X_all, y_all = [], []
        for file in os.listdir(data_dir):
            if file.endswith(".npz") and file.replace(".npz", "") in allowed_ids:
                data = np.load(os.path.join(data_dir, file))
                X_all.append(data["X"])
                y_all.append(data["y"])

        if not X_all:
            raise ValueError(f"❌ Hiçbir uygun .npz dosyası bulunamadı → {data_dir}")

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, input_channels, input_length):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_length)
            out = self.feature_extractor(dummy_input)
            self.flatten_dim = out.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x.view(-1)  # Güvenli çıkış şekli

def train_and_evaluate(data_dir, train_ids_path, combo_name):
    dataset = SegmentDataset(data_dir, train_ids_path)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = CNN1D(input_channels=dataset.X.shape[1], input_length=dataset.X.shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    final_train_loss = 0
    patience = 5
    counter = 0

    print(f"🚀 Eğitim başlatıldı → {combo_name}")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_train_loss = total_loss

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()

        print(f"📊 Epoch {epoch} - Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/{combo_name}.pt")
            print("💾 Yeni en iyi model kaydedildi")
        else:
            counter += 1
            print(f"⏸️ İyileşme yok → {counter}/{patience}")
            if counter >= patience:
                print("🛑 Early stopping → Eğitim durduruldu")
                break

    return {
        "combo_name": combo_name,
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(best_val_loss, 4),
        "epochs": epoch
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_ids", required=True)
    parser.add_argument("--combo_name", default="combo")
    args = parser.parse_args()

    result = train_and_evaluate(args.data_dir, args.train_ids, args.combo_name)
    print(result)
