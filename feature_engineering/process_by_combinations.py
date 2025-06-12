import os
import sys
import csv
import wfdb
import numpy as np
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering.segment_utils import (
    butter_bandpass_filter, normalize_segment, is_valid_segment
)
from feature_engineering.config import (
    SEGMENT_LEN, MAX_SEGMENTS, LOS_RANGE
)

COMBO_CSV = "outputs/analysis/channel_combinations.csv"
DATA_ROOT = "data/raw"
OUTPUT_ROOT = "data/processed_by_combo"
TOP_N = 5

def load_combinations(csv_path, top_n=5):
    combos = []
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= top_n:
                break
            combo_str, _ = line.strip().split(",", 1)
            combos.append(set(map(str.strip, combo_str.strip('"').split(","))))
    return combos

def process_for_combo(combo_set, combo_index):
    combo_name = f"combo_{combo_index+1}_" + "_".join(sorted(combo_set))
    output_dir = os.path.join(OUTPUT_ROOT, combo_name.replace(" ", "_"))
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(".hea") and "_layout" not in file:
                path = os.path.join(root, file).replace(".hea", "")
                try:
                    record = wfdb.rdrecord(path)
                    sig_names = record.sig_name

                    if not combo_set.issubset(set(sig_names)):
                        continue

                    signal = record.p_signal

                    # ❗️Kısa sinyal kontrolü
                    if signal.shape[0] < 30:
                        print(f"⏭️ Çok kısa sinyal atlandı: {path} ({signal.shape[0]} örnek)")
                        continue

                    full_signal = np.zeros((signal.shape[0], len(combo_set)))
                    for idx, ch in enumerate(combo_set):
                        full_signal[:, idx] = signal[:, sig_names.index(ch)]

                    full_signal = butter_bandpass_filter(full_signal)
                    segments = []
                    for i in range(signal.shape[0] // SEGMENT_LEN):
                        start = i * SEGMENT_LEN
                        end = start + SEGMENT_LEN
                        segment = full_signal[start:end, :]
                        if not is_valid_segment(segment):
                            continue
                        segment = normalize_segment(segment)
                        segments.append(segment)
                        if len(segments) >= MAX_SEGMENTS:
                            break

                    if segments:
                        los = round(np.random.uniform(*LOS_RANGE), 2)
                        los_labels = [los] * len(segments)
                        filename = os.path.basename(path)
                        out_path = os.path.join(output_dir, filename + ".npz")
                        np.savez(out_path, X=np.array(segments), y=np.array(los_labels))
                        print(f"✅ {filename} → {len(segments)} segment → {combo_name}")

                except Exception as e:
                    print(f"⚠️ Hata ({path}): {e}")

if __name__ == "__main__":
    combos = load_combinations(COMBO_CSV, TOP_N)
    for i, combo in enumerate(combos):
        process_for_combo(combo, i)
