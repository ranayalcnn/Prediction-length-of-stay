import wfdb
import numpy as np
import os
from scipy.signal import butter, filtfilt
import pandas as pd

SAMPLE_RATE = 125
SEGMENT_SEC = 30
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

los_df = pd.read_csv("data/los_labels.csv")
los_dict = dict(zip(los_df["record_id"], los_df["los"]))

def process_record(path, max_segments=100):
    record_id = os.path.basename(path).replace(".hea", "")


    if record_id not in los_dict:
        print(f"⚠️ Etiket bulunamadı: {record_id}")
        return

    try:
        record = wfdb.rdrecord(path)
        signal = record.p_signal
        signal = butter_bandpass_filter(signal)
    except Exception as e:
        print(f"⚠️ Hata ({path}): {e}")
        return

    segments = []
    for i in range(signal.shape[0] // SEGMENT_LEN):
        segment = signal[i * SEGMENT_LEN:(i + 1) * SEGMENT_LEN]
        if not is_valid_segment(segment):
            continue
        segment = normalize_segment(segment)
        segments.append(segment)
        if len(segments) >= max_segments:
            break

    if not segments:
        return

    los = los_dict[record_id]
    labels = [los] * len(segments)

    output_dir = f"data/processed_{signal.shape[1]}ch"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, record_id + ".npz")
    np.savez(output_path, X=np.array(segments), y=np.array(labels))
    print(f"✅ {record_id} → {len(segments)} segment ({signal.shape[1]} kanal)")

def process_all(input_root="data/raw"):
    for root, _, files in os.walk(input_root):  # 🔁 Alt klasörleri de tara
        for file in files:
            if file.endswith(".hea"):
                full_path = os.path.join(root, file)
                full_path = full_path.replace(".hea", "")  # wfdb dosya adı .hea içermez
                process_record(full_path)


if __name__ == "__main__":
    process_all()
