# preprocess.py

import os
import random
import numpy as np
import wfdb

from feature_engineering.config import (
    SAMPLE_RATE, SEGMENT_LEN, MAX_SEGMENTS,
    STANDARD_CHANNELS, LOS_RANGE
)
from feature_engineering.segment_utils import (
    butter_bandpass_filter,
    normalize_segment,
    is_valid_segment
)

def process_record(path, output_dir):
    try:
        record = wfdb.rdrecord(path)
        available_channels = record.sig_name
        signal = record.p_signal
    except Exception as e:
        print(f"⚠️ Okuma hatası ({path}): {e}")
        return

    channel_map = {ch: i for i, ch in enumerate(available_channels) if ch in STANDARD_CHANNELS}
    if not channel_map:
        print(f"⛔ Uygun kanal bulunamadı ({path}): {available_channels}")
        return

    full_signal = np.zeros((signal.shape[0], len(STANDARD_CHANNELS)))
    for idx, ch in enumerate(STANDARD_CHANNELS):
        if ch in channel_map:
            full_signal[:, idx] = signal[:, channel_map[ch]]

    try:
        full_signal = butter_bandpass_filter(full_signal)
    except Exception as e:
        print(f"⚠️ Filtreleme hatası ({path}): {e}")
        return

    segments = []
    total_len = full_signal.shape[0]
    num_segments = total_len // SEGMENT_LEN

    for i in range(num_segments):
        start = i * SEGMENT_LEN
        end = start + SEGMENT_LEN
        segment = full_signal[start:end, :]
        if not is_valid_segment(segment):
            continue
        segment = normalize_segment(segment)
        segments.append(segment)
        if len(segments) >= MAX_SEGMENTS:
            break

    if not segments:
        print(f"⛔ Yetersiz geçerli segment ({path})")
        return

    los = round(random.uniform(*LOS_RANGE), 2)
    los_labels = [los] * len(segments)

    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename + ".npz")
    np.savez(output_path, X=np.array(segments), y=np.array(los_labels))
    print(f"✅ {filename} → {len(segments)} segment (LOS: {los} gün)")

def process_all(input_root="data/raw", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".hea") and "_layout" not in file:
                full_path = os.path.join(root, file).replace(".hea", "")
                process_record(full_path, output_dir)

if __name__ == "__main__":
    process_all()
