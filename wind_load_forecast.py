import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_preprocess_data(data_path, sequence_length=24, train_split=0.8):
    # Load data (example with synthetic data for demonstration)
    # In practice, replace this with your actual data loading logic
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    wind_speed = np.sin(time/50) + np.random.normal(0, 0.1, n_samples)
    load = 2 * np.sin(time/50) + np.random.normal(0, 0.2, n_samples)
    
    # Combine features
    data = np.column_stack((wind_speed, load))
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:(i + sequence_length)])
        y.append(data_scaled[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def create_cnn_lstm_model(sequence_length, n_features):
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, n_features)),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        # LSTM layers for temporal dependencies
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(30),
        Dropout(0.2),
        
        # Dense layers for prediction
        Dense(16, activation='relu'),
        Dense(n_features)  # Output layer (wind speed and load)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the trained model
    model_path = os.path.join(results_dir, 'model.keras')
    save_model(model, model_path)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save predictions and metrics
    np.save(os.path.join(results_dir, 'predictions.npy'), y_pred)
    metrics = pd.DataFrame({
        'Metric': ['MSE', 'MAE'],
        'Value': [mse, mae]
    })
    metrics.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    print(f'Test MSE: {mse:.4f}')
    print(f'Test MAE: {mae:.4f}')
    print(f'Results saved in: {results_dir}')
    
    return history, y_pred

def plot_results(history, y_test, y_pred, results_dir):
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.plot(y_test[:, 0], label='Actual Wind')
    plt.plot(y_pred[:, 0], label='Predicted Wind')
    plt.title('Wind Speed Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Normalized Wind Speed')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'results_plot.png'))
    plt.close()

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        data_path=None,  # Replace with your data path
        sequence_length=24
    )
    
    # Create and train model
    model = create_cnn_lstm_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2]
    )
    
    # Train and evaluate
    history, y_pred = train_and_evaluate_model(
        model, X_train, X_test, y_train, y_test
    )
    
    # Get the results directory path
    results_dir = max([d for d in os.listdir() if d.startswith('results_')], key=os.path.getctime)
    
    # Plot and save results
    plot_results(history, y_test, y_pred, results_dir)

if __name__ == '__main__':
    main()