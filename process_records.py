import wfdb
import numpy as np
import os
from scipy.signal import butter, filtfilt
import pandas as pd

# Ayarlar
SAMPLE_RATE = 125
SEGMENT_SEC = 30
SEGMENT_LEN = SAMPLE_RATE * SEGMENT_SEC
MAX_SEGMENTS = 100

# Bandpass filtre
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=SAMPLE_RATE, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Z-score normalize
def normalize_segment(segment):
    return (segment - np.mean(segment, axis=0)) / (np.std(segment, axis=0) + 1e-8)

# Segmentin geçerliliğini kontrol et
def is_valid_segment(segment):
    return not np.isnan(segment).any() and not np.all(segment == 0)

# Etiketleri CSV'den yükle
def load_los_labels(csv_path="data/los_labels.csv"):
    df = pd.read_csv(csv_path)
    return dict(zip(df["record_id"], df["los"]))

# Tek bir kaydı işle
def process_record(path, los_dict):
    record_id = os.path.basename(path).replace(".hea", "")
    if record_id not in los_dict:
        print(f"⚠️ Etiket bulunamadı: {record_id}")
        return

    try:
        record = wfdb.rdrecord(path)
        signal = record.p_signal
        signal = butter_bandpass_filter(signal)
    except Exception as e:
        print(f"⚠️ {record_id} kaydı okunamadı: {e}")
        return

    segments = []
    for i in range(signal.shape[0] // SEGMENT_LEN):
        segment = signal[i * SEGMENT_LEN:(i + 1) * SEGMENT_LEN]
        if not is_valid_segment(segment):
            continue
        segment = normalize_segment(segment)
        segments.append(segment)
        if len(segments) >= MAX_SEGMENTS:
            break

    if not segments:
        print(f"⚠️ {record_id} için geçerli segment bulunamadı.")
        return

    los = los_dict[record_id]
    labels = [los] * len(segments)

    output_dir = f"data/processed_{signal.shape[1]}ch"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, record_id + ".npz")
    np.savez(output_path, X=np.array(segments), y=np.array(labels))
    print(f"✅ {record_id}: {len(segments)} segment işlendi ({signal.shape[1]} kanal)")

# Tüm kayıtları tara ve işle
def process_all(input_root="data/raw", los_csv="data/los_labels.csv"):
    los_dict = load_los_labels(los_csv)
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(".hea"):
                full_path = os.path.join(root, file[:-4])  # .hea'siz yol
                process_record(full_path, los_dict)

# Ana çalıştırma
if __name__ == "__main__":
    process_all()
