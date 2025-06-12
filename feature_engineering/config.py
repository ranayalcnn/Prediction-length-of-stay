# config.py – Tüm projede ortak kullanılan sabitler

# Segment ayarları
SAMPLE_RATE = 125           # 125 Hz örnekleme hızı
SEGMENT_SECONDS = 30        # 30 saniyelik segmentler
SEGMENT_LEN = SAMPLE_RATE * SEGMENT_SECONDS

# Kullanılacak standart kanal sırası (eksikler 0'la doldurulacak)
STANDARD_CHANNELS = [
    'II', 'V', 'ABP', 'PLETH', 'RESP', 'CVP', 'I', 'III', 'AVR'
]

# Maksimum segment sayısı (her kayıttan)
MAX_SEGMENTS = 100

# Band-pass filtre parametreleri
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
FILTER_ORDER = 4

# Rastgele etiketleme (şimdilik dummy LOS)
LOS_RANGE = (1.0, 10.0)  # gün cinsinden
