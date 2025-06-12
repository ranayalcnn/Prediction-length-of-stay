import numpy as np
from scipy.signal import butter, filtfilt
from feature_engineering.config import (
    FILTER_LOW, FILTER_HIGH, FILTER_ORDER, SAMPLE_RATE
)

def butter_bandpass_filter(data, lowcut=FILTER_LOW, highcut=FILTER_HIGH, fs=SAMPLE_RATE, order=FILTER_ORDER):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def normalize_segment(segment):
    return (segment - np.mean(segment, axis=0)) / (np.std(segment, axis=0) + 1e-8)

def is_valid_segment(segment):
    return not np.isnan(segment).any() and not np.all(segment == 0)
