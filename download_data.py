import os
import wfdb
import time
from collections import defaultdict

def build_balanced_record_list(input_file='records.txt', count_per_group=4, existing_ids=None, max_per_patient_id=30):
    with open(input_file, 'r') as f:
        all_records = f.read().splitlines()

    groups = {str(i): [] for i in range(30, 40)}  # sadece 32–39 klasörleri
    patient_count = defaultdict(int)  # her hasta ID için sayım
    groups_filled = {g: [] for g in groups}

    for line in all_records:
        parts = line.strip().split("/")
        if len(parts) < 3:
            continue

        group = parts[0]                   # örn: "32"
        rec = parts[-1]                    # örn: "3200013_0007"
        patient_id = rec.split("_")[0]     # örn: "3200013"
        full_path = "/".join(parts)

        if group in groups:
            if existing_ids and rec in existing_ids:
                continue
            if patient_count[patient_id] >= max_per_patient_id:
                continue  # 🛑 hasta limiti aşıldı
            if len(groups_filled[group]) < count_per_group:
                groups_filled[group].append(full_path)
                patient_count[patient_id] += 1

    selected = []
    for group in groups:
        selected.extend(groups_filled[group])

    print(f"\n🎯 {len(selected)} yeni kayıt seçildi (her klasörden {count_per_group}, her hastadan max {max_per_patient_id})")

    print("\n📊 Hasta başına alınan kayıt sayısı:")
    for pid, count in sorted(patient_count.items()):
        print(f"• {pid}: {count}")

    return selected

def get_existing_record_ids(data_root="data/raw"):
    ids = set()
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".hea"):
                ids.add(file.replace(".hea", ""))
    return ids

def download_records(records):
    print("\n📥 İndirme başlıyor...\n")
    start_time = time.time()

    for full_record_path in records:
        rec = full_record_path.split("/")[-1]
        group = full_record_path.split("/")[0]
        folder = os.path.join("data/raw", group, rec)
        hea_file = os.path.join(folder, f"{rec}.hea")

        if os.path.exists(hea_file):
            print(f"⏭️ {rec} zaten var, atlandı.")
            continue

        print(f"🔽 {full_record_path}")
        try:
            wfdb.dl_database(
                db_dir='mimic3wdb',
                dl_dir='data/raw',
                records=[full_record_path]
            )
        except Exception as e:
            print(f"⚠️ Hata ({full_record_path}): {e}")

    elapsed = round(time.time() - start_time, 2)
    print(f"\n✅ İndirme tamamlandı. ⏱️ Toplam süre: {elapsed} saniye")

if __name__ == "__main__":
    if not os.path.exists('records.txt'):
        print("❌ HATA: records.txt bulunamadı.")
        print("👉 https://physionet.org/static/published-projects/mimic3wdb/mimic3wdb-1.0/matched/RECORDS")
        exit()

    existing_ids = get_existing_record_ids()
    print(f"📦 Tespit edilen mevcut kayıt sayısı: {len(existing_ids)}")

    new_records = build_balanced_record_list(
        input_file='records.txt',
        count_per_group=4,            # klasör başına en fazla 4 kayıt
        existing_ids=existing_ids,
        max_per_patient_id=30         # hasta başına en fazla 30 kayıt
    )

    download_records(new_records)
