# Wind Power Forecasting System

A sophisticated forecasting system that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to predict wind speed and load patterns. This hybrid model leverages both spatial and temporal features for accurate predictions.

## Project Overview

This project implements a deep learning model that:
- Processes historical wind speed and load data
- Uses a CNN-LSTM hybrid architecture for feature extraction and temporal pattern recognition
- Provides comprehensive visualizations of predictions and patterns
- Saves model results and visualizations for analysis

## Features

- **Hybrid Model Architecture**: Combines CNN layers for feature extraction and LSTM layers for temporal dependency learning
- **Data Preprocessing**: Includes sequence creation and normalization
- **Comprehensive Visualization**: Multiple visualization types including:
  - Wind speed comparisons
  - Daily patterns
  - Training history plots
- **Model Persistence**: Automatically saves trained models and predictions
- **Performance Metrics**: Tracks and saves MSE and MAE metrics

## Installation

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

## Usage

1. Prepare your wind speed and load data
2. Run the main training script:
   ```bash
   python wind_load_forecast.py
   ```
3. View results in the automatically created `results_[timestamp]` directory
4. Generate visualizations:
   ```bash
   python wind_power_visualization.py
   ```

## Model Architecture

### CNN Layers
- First Conv1D: 64 filters, kernel size 3
- Second Conv1D: 32 filters, kernel size 3
- Batch Normalization after each Conv1D

### LSTM Layers
- First LSTM: 50 units with return sequences
- Second LSTM: 30 units
- Dropout layers (0.2) after each LSTM

### Dense Layers
- Hidden Dense: 16 units with ReLU
- Output Dense: Predicts wind speed and load

## Output Directory Structure

After running the model, a `results_[timestamp]` directory is created containing:
- `model.keras`: Saved trained model
- `predictions.npy`: Model predictions
- `metrics.csv`: Performance metrics
- `training_history.csv`: Training progress data
- Various visualization plots

## Visualization Types

1. **Wind Speed Comparison**: Actual vs predicted wind speeds
2. **Daily Pattern**: Average wind power patterns over 24-hour periods
3. **Training History**: Loss and metrics during training

## Performance Metrics

The system tracks:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Metrics are saved in the results directory for each training run.

## Contributing

Feel free to open issues or submit pull requests for improvements.
