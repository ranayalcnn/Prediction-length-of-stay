import wfdb
import os
import csv
from collections import Counter
import matplotlib.pyplot as plt

# 📁 Klasör yapılarını oluştur
os.makedirs("outputs/analysis", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

input_root = "data/raw"
channel_counter = Counter()
combo_counter = Counter()
unique_channel_sets = set()

# 🔍 .hea dosyalarını tara
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(".hea") and "_layout" not in file:
            path = os.path.join(root, file).replace(".hea", "")
            try:
                record = wfdb.rdrecord(path)
                channels = tuple(record.sig_name)
                unique_channel_sets.add(channels)
                channel_counter.update(channels)
                combo_counter[tuple(channels)] += 1
            except Exception as e:
                print(f"⚠️ Hata ({file}): {e}")

# 📊 En sık geçen 20 kanal
top_channels = channel_counter.most_common(20)
names, counts = zip(*top_channels)

# 🖼️ Grafik oluştur ve kaydet
plot_path = "outputs/plots/channel_distribution.png"
plt.figure(figsize=(12, 6))
plt.barh(names[::-1], counts[::-1])
plt.title("En Sık Kullanılan 20 Kanal")
plt.xlabel("Kayıt Sayısı")
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

# 📄 Kanal sayımı CSV
csv_path = "outputs/analysis/channel_counts.csv"
with open(csv_path, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Kanal", "Kayıt Sayısı"])
    for name, count in channel_counter.most_common():
        writer.writerow([name, count])

# 📄 Kombinasyon CSV
combo_csv_path = "outputs/analysis/channel_combinations.csv"
with open(combo_csv_path, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Kanal Kombinasyonu", "Kayıt Sayısı"])
    for combo, count in combo_counter.most_common():
        combo_str = ", ".join(combo)
        writer.writerow([combo_str, count])

print(f"✅ CSV oluşturuldu: {csv_path}")
print(f"✅ Kanal kombinasyonları kaydedildi: {combo_csv_path}")
print(f"✅ Grafik kaydedildi: {plot_path}")
