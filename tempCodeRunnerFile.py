import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from glob import glob
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.deep_cnn import DeepCNN
from models.resnet_1d import ResNet1D
from models.inception_1d import Inception1D
from models.custom_cnn import CustomCNN

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Kullanılan cihaz:", device)

# Sabitler
DATA_ROOT = "data"
TARGET_CHANNELS = 8
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALID_RATIO = 0.2
TEST_RATIO = 0.1

def load_all_data(root_dir, target_channels=TARGET_CHANNELS):
    X, y = [], []
    npz_files = glob(os.path.join(root_dir, "processed_*ch", "*.npz"))
    for path in npz_files:
        data = np.load(path)
        x = data["X"]
        if x.shape[2] > target_channels:
            continue
        padded = np.zeros((x.shape[0], x.shape[1], target_channels))
        padded[:, :, :x.shape[2]] = x
        X.append(padded)
        y.append(data["y"])
    if not X:
        raise ValueError("Hiç uygun veri bulunamadı.")
    return np.vstack(X), np.hstack(y)

def augment_batch(batch_X, noise_std=0.05, time_shift=40, scale_range=(0.7, 1.3)):
    batch = batch_X.clone()
    noise = torch.randn_like(batch) * noise_std
    batch += noise
    shift = torch.randint(-time_shift, time_shift + 1, (batch.size(0),))
    for i in range(batch.size(0)):
        batch[i] = torch.roll(batch[i], shifts=int(shift[i]), dims=1)
    scales = torch.empty(batch.size(0), 1, 1).uniform_(*scale_range).to(batch.device)
    batch *= scales
    return batch

def train_model(model, train_loader, val_loader, model_name, patience=5):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = augment_batch(batch_X)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{model_name}] Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"model/{model_name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f"model/{model_name}.pt"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze().cpu().numpy()
            labels = batch_y.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    acc = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)) <= 1.0)

    print(f"\n📊 [{model_name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f} | Accuracy (±1 gün): {acc*100:.2f}%")
    return mae

def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze().cpu().numpy()
            labels = batch_y.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    errors = all_preds - all_labels

    plt.figure(figsize=(7, 5))
    plt.scatter(all_labels, all_preds, alpha=0.5, color="royalblue")
    plt.plot([min(all_labels), max(all_labels)],
             [min(all_labels), max(all_labels)],
             'r--', label='y = x')
    plt.xlabel("Gerçek LOS (gün)")
    plt.ylabel("Tahmin LOS (gün)")
    plt.title("📊 Gerçek vs Tahmin LOS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Tahmin Hatası (Tahmin - Gerçek)")
    plt.ylabel("Segment Sayısı")
    plt.title("📉 Hata Dağılımı")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    acc = np.mean(np.abs(errors) <= 1.0)

    print(f"📌 Test Sonuçları\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f} | Accuracy (±1 gün): {acc*100:.2f}%")

if __name__ == "__main__":
    X, y = load_all_data(DATA_ROOT)
    X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    total_len = len(dataset)
    test_len = int(TEST_RATIO * total_len)
    val_len = int(VALID_RATIO * total_len)
    train_len = total_len - test_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    print(f"📊 Toplam örnek sayısı: {len(dataset)}")
    print(f"📂 Eğitim seti: {len(train_set)} örnek")
    print(f"🧪 Doğrulama seti: {len(val_set)} örnek")
    print(f"🧾 Test seti: {len(test_set)} örnek")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    os.makedirs("model", exist_ok=True)

    model_dict = {
        "deep_cnn": DeepCNN(in_channels=TARGET_CHANNELS),
        "resnet_1d": ResNet1D(in_channels=TARGET_CHANNELS),
        "inception_1d": Inception1D(in_channels=TARGET_CHANNELS),
        "custom_cnn": CustomCNN(in_channels=TARGET_CHANNELS)
    }

    results = {}
    for name, model in model_dict.items():
        print(f"\n🚀 {name.upper()} eğitimi başlıyor...")
        mae = train_model(model, train_loader, val_loader, model_name=name)
        results[name] = mae

    best_model = min(results, key=results.get)
    print(f"\n🎯 En iyi model: {best_model.upper()} (MAE: {results[best_model]:.2f})")

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    best_model_path = f"model/{best_model}.pt"
    model = model_dict[best_model]
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    evaluate_on_test(model, test_loader)

    print("\n📊 Tüm modeller test seti üzerinde değerlendiriliyor...\n")
    for name in model_dict:
        print(f"\n📈 {name.upper()} modeli test setinde değerlendiriliyor...")
        model = model_dict[name]
        model.load_state_dict(torch.load(f"model/{name}.pt"))
        model.to(device)
        evaluate_on_test(model, test_loader)