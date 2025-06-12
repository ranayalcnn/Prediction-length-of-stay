import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from glob import glob
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F

from models.deep_cnn import DeepCNN
from models.resnet_1d import ResNet1D
from models.inception_1d import Inception1D
from models.custom_cnn import CustomCNN

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Kullanılan cihaz:", device)

# Sabitler
DATA_ROOT = "data"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
VALID_RATIO = 0.2
TEST_RATIO = 0.1

def load_data_for_channels(root_dir, channels):
    """Belirli kanal sayısı için veriyi yükle"""
    X, y = [], []
    npz_files = glob(os.path.join(root_dir, f"processed_{channels}ch", "*.npz"))
    for path in npz_files:
        data = np.load(path)
        x = data["X"]
        if x.shape[2] != channels:
            continue
        X.append(x)
        y.append(data["y"])
    if not X:
        raise ValueError(f"{channels} kanallı veri bulunamadı.")
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

def train_model(model, train_loader, val_loader, model_name, in_channels, patience=5):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

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
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{model_name}] Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f"model/{model_name}_{in_channels}ch.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # En iyi modeli yükle
    checkpoint = torch.load(f"model/{model_name}_{in_channels}ch.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test seti üzerinde değerlendirme
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
    return mae, train_losses, val_losses

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

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    acc = np.mean(np.abs(errors) <= 1.0)

    print(f"📌 Test Sonuçları\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f} | Accuracy (±1 gün): {acc*100:.2f}%")
    return mae, rmse, r2, acc

def train_for_channels(channels):
    print(f"\n🚀 {channels} kanallı veriler için eğitim başlıyor...")
    
    try:
        # Veriyi yükle
        X, y = load_data_for_channels(DATA_ROOT, channels)
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
        print(f"🗞 Test seti: {len(test_set)} örnek")

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
        os.makedirs("model", exist_ok=True)

        model_dict = {
            "deep_cnn": DeepCNN(in_channels=channels),
            "resnet_1d": ResNet1D(in_channels=channels),
            "inception_1d": Inception1D(in_channels=channels),
            "custom_cnn": CustomCNN(in_channels=channels)
        }

        results = {}
        for name, model in model_dict.items():
            print(f"\n🚀 {name.upper()} eğitimi başlıyor...")
            mae, train_losses, val_losses = train_model(model, train_loader, val_loader, model_name=name, in_channels=channels)
            results[name] = mae

        best_model = min(results, key=results.get)
        print(f"\n🎯 En iyi model: {best_model.upper()} (MAE: {results[best_model]:.2f})")

        # Test seti üzerinde değerlendirme
        best_model_path = f"model/{best_model}_{channels}ch.pt"
        model = model_dict[best_model]
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        mae, rmse, r2, acc = evaluate_on_test(model, test_loader)
        
        # Sonuçları kaydet
        with open(f"outputs/results_{channels}ch.txt", "w") as f:
            f.write(f"Best Model: {best_model}\n")
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
            f.write(f"R²: {r2:.3f}\n")
            f.write(f"Accuracy: {acc*100:.2f}%\n")
            
    except Exception as e:
        print(f"❌ {channels} kanallı veriler için eğitim başarısız: {str(e)}")

if __name__ == "__main__":
    # Her kanal sayısı için model eğitimi
    for channels in range(1, 9):  # 1'den 8'e kadar tüm kanal sayıları
        train_for_channels(channels)