import os
import csv
import wfdb
import numpy as np

def infer_los_from_signal(signal):
    # Çok uzun kayıtları sınırla: ilk 5 dakika (125Hz * 60 * 5)
    max_samples = 125 * 60 * 5
    if signal.shape[0] > max_samples:
        signal = signal[:max_samples]

    energy = np.sum(np.square(signal))     # toplam enerji
    var = np.var(signal)                   # varyans
    score = (energy * var) / 1e6           # kombine skor
    los = np.clip(score, 1.0, 10.0)        # 1–10 aralığına sınırla
    return round(los, 2)

def create_los_csv(waveform_dir="data/raw", output_csv="data/los_labels.csv"):
    entries = []
    for root, _, files in os.walk(waveform_dir):
        for file in files:
            if file.endswith(".hea"):
                record_id = file.replace(".hea", "")
                try:
                    record_path = os.path.join(root, record_id)
                    record = wfdb.rdrecord(record_path)
                    signal = record.p_signal
                    los = infer_los_from_signal(signal)
                    entries.append((record_id, los))
                except Exception as e:
                    print(f"⚠️ {record_id} okunamadı: {e}")

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "los"])
        writer.writerows(entries)

    print(f"✅ Etiket dosyası oluşturuldu: {output_csv} ({len(entries)} kayıt)")

if __name__ == "__main__":
    create_los_csv()
