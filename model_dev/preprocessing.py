import wfdb
import numpy as np
import os
from scipy.signal import butter, filtfilt
import random

SAMPLE_RATE = 125  # genellikle 125 Hz
SEGMENT_SEC = 30   # 30 saniyelik segmentler
SEGMENT_LEN = SAMPLE_RATE * SEGMENT_SEC

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=125, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def normalize_segment(segment):
    return (segment - np.mean(segment, axis=0)) / (np.std(segment, axis=0) + 1e-8)

def is_valid_segment(segment):
    return not np.isnan(segment).any() and not np.all(segment == 0)

def process_record(path, output_dir, max_segments=100):
    try:
        record = wfdb.rdrecord(path)
        signal = record.p_signal  # (zaman, kanal)
    except Exception as e:
        print(f"⚠️ Okuma hatası ({path}): {e}")
        return

    # Filtre uygula
    try:
        signal = butter_bandpass_filter(signal)
    except Exception as e:
        print(f"⚠️ Filtreleme hatası ({path}): {e}")
        return

    segments = []
    total_len = signal.shape[0]
    num_segments = total_len // SEGMENT_LEN

    for i in range(num_segments):
        start = i * SEGMENT_LEN
        end = start + SEGMENT_LEN
        segment = signal[start:end, :]
        if not is_valid_segment(segment):
            continue
        segment = normalize_segment(segment)
        segments.append(segment)

        if len(segments) >= max_segments:
            break  # çok uzun sinyallerde aşırı örnekleme engellenir

    if not segments:
        print(f"⛔ Yetersiz/boş sinyal: {path}")
        return

    # Dummy LOS etiketi ver (örnek: 1.0–10.0 gün arası rastgele)
    los = round(random.uniform(1.0, 10.0), 2)
    los_labels = [los] * len(segments)

    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename + ".npz")

    np.savez(output_path, X=np.array(segments), y=np.array(los_labels))
    print(f"✅ {filename} → {len(segments)} segment (LOS: {los} gün)")

def process_all(input_root="data/raw", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".hea"):
                full_path = os.path.join(root, file).replace(".hea", "")
                process_record(full_path, output_dir)
                
for file in os.listdir("data/processed"):
    if file.endswith(".npz"):
        data = np.load(os.path.join("data/processed", file))
        print(file, "->", data["X"].shape)

if __name__ == "__main__":
    process_all()
