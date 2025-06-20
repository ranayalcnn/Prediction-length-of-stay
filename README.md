# ICU Length of Stay Prediction Using Waveform Data

## Project Description

This project aims to predict the length of stay (LOS) in the Intensive Care Unit (ICU) using raw physiological waveform data from the MIMIC-III Waveform Database. Traditional models rely on Electronic Health Records (EHR), but this project leverages deep learning techniques directly on high-resolution signals like ECG, ABP, and PPG to uncover temporal patterns that might improve prediction accuracy and real-time ICU monitoring.

## Dataset

**MIMIC-III Waveform Database v1.0**

* Collected from ICU patients at the Beth Israel Deaconess Medical Center.
* Includes ECG, arterial blood pressure, photoplethysmography, and derived signals.
* Long, continuous, and unprocessed high-resolution waveforms.
* [Access the dataset](https://physionet.org/content/mimic3wdb/1.0/)

## Objectives

* Predict ICU length of stay using time-series waveform data.
* Develop and compare deep learning models (CNN, LSTM, Transformers).
* Improve interpretability using visualization techniques like attention maps.

## Preprocessing Steps

* Segmentation: Slice waveforms into fixed-length windows.
* Denoising: Apply signal filtering to remove artifacts.
* Normalization: Scale signals for better neural network performance.
* Labeling: Map waveform segments to corresponding LOS labels using admission/discharge times.

## Models Used

* Convolutional Neural Networks (CNNs)
* Long Short-Term Memory networks (LSTMs)
* Transformer-based architectures
* Hyperparameter tuning (e.g., learning rate, window size, layer depth)

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)
* Time-series cross-validation for robustness

## Interpretability

* Use Grad-CAM or attention weights to highlight important waveform regions.
* Provide visual insight into physiological dynamics linked to extended ICU stays.

## Challenges

* Noise: Raw waveforms require extensive cleaning.
* Label Alignment: Mapping LOS with time windows is non-trivial.
* Compute: High-resolution data demands significant resources.

## How to Download the Dataset

1. **Request Access**  
   Go to [PhysioNet Credentialing](https://physionet.org/works/MIMICIIIClinicalDatabase/)  
   - Create an account  
   - Complete CITI training  
   - Accept the data use agreement for MIMIC-III Waveform

2. **Get Matched Records List**  
   Download the records file from:  
   [https://physionet.org/static/published-projects/mimic3wdb/mimic3wdb-1.0/matched/RECORDS](https://physionet.org/static/published-projects/mimic3wdb/mimic3wdb-1.0/matched/RECORDS)  
   Save it as `records.txt` in the project root.

3. **Download Data (30–39 folders)**  
   Run the script below to download a balanced subset:

   ```bash
   python download_records.py

## How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/ranayalcnn/Prediction-length-of-stay.git
   cd Prediction-length-of-stay
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run preprocessing:

   ```bash
   python preprocessing.py
   ```

4. Train model:

   ```bash
   python train.py
   ```

5. Evaluate:

   ```bash
   python evaluate.py
   ```

## License

This project is licensed under the MIT License.
