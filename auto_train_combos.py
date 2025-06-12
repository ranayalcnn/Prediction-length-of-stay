import os
import csv
from model_dev.training import train_and_evaluate

SPLIT_DIR = "splits"
DATA_DIR = "data/processed_by_combo"
OUTPUT_CSV = "outputs/results.csv"

def main():
    results = []

    for fname in os.listdir(SPLIT_DIR):
        if not fname.endswith("_train_ids.txt"):
            continue

        combo_name = fname.replace("_train_ids.txt", "")
        train_ids_path = os.path.join(SPLIT_DIR, fname)
        data_path = os.path.join(DATA_DIR, combo_name)

        if not os.path.exists(data_path):
            print(f"⛔ Atlandı: {combo_name} için veri klasörü bulunamadı.")
            continue

        print(f"🚀 Eğitim başlıyor: {combo_name}")
        try:
            result = train_and_evaluate(data_path, train_ids_path, combo_name)
            results.append(result)
            print(f"✅ Tamamlandı: {combo_name}")
        except Exception as e:
            print(f"⚠️ Hata: {combo_name} → {e}")

    # CSV'ye yaz
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["combo_name", "train_loss", "val_loss", "epochs"])
        writer.writeheader()
        writer.writerows(results)

    print(f"📁 Sonuçlar yazıldı: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
