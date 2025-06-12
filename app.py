import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import torch
import time
import json
from models.resnet_1d import ResNet1D
from models.deep_cnn import DeepCNN
from models.inception_1d import Inception1D
from models.custom_cnn import CustomCNN

# Set page config
st.set_page_config(
    page_title="ICU Length of Stay Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("ICU Length of Stay Prediction")
st.markdown("""
This application predicts ICU Length of Stay (LOS) using physiological waveform data from the MIMIC-III database.
The model analyzes continuous waveforms such as ECG, ABP, and PPG to provide accurate predictions.
""")

# Sidebar for project information
with st.sidebar:
    st.header("Project Information")
    st.markdown("""
    ### About
    This project utilizes deep learning to predict ICU Length of Stay using raw physiological signals.
    
    ### Model
    - Multiple models trained for different channel counts (1-8)
    - Best model selected based on MAE
    - Input: Physiological signals (1-8 channels)
    - Output: Predicted Length of Stay in days
    
    ### Dataset
    - MIMIC-III Waveform Database
    - Contains ECG, ABP, PPG signals
    - High-resolution physiological data
    """)

def load_model_metrics(channels):
    """Load model metrics from the outputs directory"""
    try:
        with open(f"outputs/results_{channels}ch.txt", "r") as f:
            lines = f.readlines()
            metrics = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":")
                    metrics[key.strip()] = value.strip()
            return metrics
    except:
        return None

def load_best_model(channels):
    """Load the best model for the given channel count"""
    try:
        metrics = load_model_metrics(channels)
        if not metrics:
            return None, None
        
        model_name = metrics.get("Best Model", "").lower()
        if model_name == "resnet_1d":
            model = ResNet1D(in_channels=channels)
        elif model_name == "deep_cnn":
            model = DeepCNN(in_channels=channels)
        elif model_name == "inception_1d":
            model = Inception1D(in_channels=channels)
        elif model_name == "custom_cnn":
            model = CustomCNN(in_channels=channels)
        else:
            return None, None
            
        model_path = f"model/{model_name}_{channels}ch.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, metrics
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
    return None, None

# Main content area
st.header("Data Upload and Analysis")

# File upload in a centered container
upload_container = st.container()
with upload_container:
    uploaded_file = st.file_uploader(
        "Upload .npz file",
        type=['npz'],
        help="Upload preprocessed waveform data in .npz format"
    )

if uploaded_file is not None and st.button("Predict", use_container_width=True):
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_upload_{int(time.time())}.npz"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load and process the data
        data = np.load(temp_path)
        X = torch.tensor(data["X"], dtype=torch.float32).permute(0, 2, 1)
        in_channels = X.shape[1]
        
        # Create tabs for main sections
        main_tab1, main_tab2 = st.tabs(["Signal Analysis", "Prediction Results"])
        
        with main_tab1:
            st.subheader("Signal Visualization")
            signal_data = X[0].numpy()  # Get first sample for visualization
            time_points = np.arange(signal_data.shape[1])
            
            # Create a figure for each channel
            for channel_idx in range(in_channels):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=signal_data[channel_idx],
                    name=f'Channel {channel_idx + 1}',
                    line=dict(width=1)
                ))
                fig.update_layout(
                    title=f'Channel {channel_idx + 1} Signal',
                    xaxis_title='Time Points',
                    yaxis_title='Amplitude',
                    showlegend=True,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with main_tab2:
            # Load the best model for this channel count
            model, metrics = load_best_model(in_channels)
            
            if model is None:
                st.error(f"No trained model found for {in_channels} channels. Please train the model first.")
            else:
                # Make prediction
                with torch.no_grad():
                    preds = model(X).squeeze().numpy()
                    mean_pred = np.mean(preds)
                
                # Display prediction in a prominent way
                st.markdown("---")
                st.markdown("### Prediction Results")
                pred_col1, pred_col2, pred_col3 = st.columns([1,2,1])
                with pred_col2:
                    st.metric(
                        label="Predicted Length of Stay",
                        value=f"{mean_pred:.2f} days",
                        delta=None
                    )
                st.markdown("---")
                
                # Display metrics in a balanced way
                st.markdown("### Model Performance")
                metrics_col1, metrics_col2, metrics_col3 = st.columns([1,2,1])
                with metrics_col2:
                    st.markdown("""
                    <div style='text-align: center; padding: 20px; background-color: #262730; border-radius: 10px; border: 1px solid #4B4B4B;'>
                        <h3 style='margin-bottom: 20px; color: #FFFFFF;'>Performance Metrics</h3>
                        <div style='font-size: 24px; margin: 15px 0; color: #FFFFFF;'>
                            MAE: <span style='color: #00FF9D; font-weight: bold;'>{}</span> days
                        </div>
                        <div style='font-size: 24px; margin: 15px 0; color: #FFFFFF;'>
                            RMSE: <span style='color: #00FF9D; font-weight: bold;'>{}</span> days
                        </div>
                        <div style='font-size: 24px; margin: 15px 0; color: #FFFFFF;'>
                            R²: <span style='color: #00FF9D; font-weight: bold;'>{}</span>
                        </div>
                    </div>
                    """.format(
                        metrics.get('MAE', 'N/A'),
                        metrics.get('RMSE', 'N/A'),
                        metrics.get('R²', 'N/A')
                    ), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass  # Ignore cleanup errors

# Visualization section
st.header("Model Performance Visualization")

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["Training Progress", "LOS Distribution"])

with tab1:
    # Training/Validation Loss Plot
    st.subheader("Training and Validation Loss")
    # Load your training history data here
    # This is a placeholder - you'll need to load your actual data
    epochs = list(range(1, 101))  # Convert range to list
    train_loss = [0.5 * np.exp(-0.05 * x) for x in epochs]
    val_loss = [0.6 * np.exp(-0.04 * x) for x in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss'))
    fig.update_layout(title='Model Training Progress',
                     xaxis_title='Epochs',
                     yaxis_title='Loss')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # LOS Distribution Plot
    st.subheader("Length of Stay Distribution")
    # This is a placeholder - you'll need to load your actual data
    los_data = np.random.gamma(2, 2, 1000)  # Example distribution
    fig = px.histogram(los_data, 
                      title='Distribution of Length of Stay',
                      labels={'value': 'Length of Stay (days)'})
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>SE 3508 Introduction to Artificial Intelligence</p>
    <p>RANA YALÇIN 220717030</p>
</div>
""", unsafe_allow_html=True)
