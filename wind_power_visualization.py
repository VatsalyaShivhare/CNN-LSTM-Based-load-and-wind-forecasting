import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_training_history(results_dir):
    history_path = os.path.join(results_dir, 'training_history.csv')
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    return None

def load_predictions(results_dir):
    pred_path = os.path.join(results_dir, 'predictions.npy')
    if os.path.exists(pred_path):
        predictions = np.load(pred_path)
        # Ensure array is 1D
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        return predictions, None
    return None, None

def plot_training_history(history_df, save_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'results_plot.png'))
    plt.close()

def plot_daily_pattern(predictions, save_dir):
    try:
        # Handle predictions of any size by taking the first 24*n hours
        n_days = len(predictions) // 24
        if n_days > 0:
            usable_predictions = predictions[:n_days*24]
            daily_avg = np.reshape(usable_predictions, (n_days, 24)).mean(axis=0)
            hours = np.arange(24)
            
            plt.figure(figsize=(12, 6))
            plt.plot(hours, daily_avg, marker='o')
            plt.title('Average Daily Wind Power Pattern')
            plt.xlabel('Hour of Day')
            plt.ylabel('Wind Power')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'daily_pattern.png'))
            plt.close()
    except Exception as e:
        print(f'Error in plot_daily_pattern: {e}')

def plot_hourly_heatmap(predictions, save_dir):
    try:
        # Handle predictions of any size by taking the first 24*n hours
        n_days = len(predictions) // 24
        if n_days > 0:
            usable_predictions = predictions[:n_days*24]
            daily_data = np.reshape(usable_predictions, (n_days, 24))
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(daily_data, cmap='YlOrRd', xticklabels=range(24))
            plt.title('Hourly Wind Power Heatmap')
            plt.xlabel('Hour of Day')
            plt.ylabel('Day')
            plt.savefig(os.path.join(save_dir, 'hourly_heatmap.png'))
            plt.close()
    except Exception as e:
        print(f'Error in plot_hourly_heatmap: {e}')

def plot_monthly_boxplot(predictions, save_dir):
    try:
        # Handle predictions of any size
        days = len(predictions) // 24
        if days >= 30:  # Only plot if we have at least one month of data
            months = min(days // 30, 12)  # Use up to 12 months of data
            
            monthly_data = []
            for i in range(months):
                start_idx = i * 30 * 24
                end_idx = min(start_idx + (30 * 24), len(predictions))
                if end_idx > start_idx:
                    monthly_data.append(predictions[start_idx:end_idx])
            
            if monthly_data:
                plt.figure(figsize=(12, 6))
                plt.boxplot(monthly_data)
                plt.title('Monthly Wind Power Distribution')
                plt.xlabel('Month')
                plt.ylabel('Wind Power')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, 'monthly_boxplot.png'))
                plt.close()
    except Exception as e:
        print(f'Error in plot_monthly_boxplot: {e}')

def create_visualizations(results_dir):
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load training history and predictions
    history_df = load_training_history(results_dir)
    predictions, actual_wind = load_predictions(results_dir)
    
    if history_df is not None:
        plot_training_history(history_df, 'results')
        print('Training history plot saved.')
    
    if predictions is not None:
        plot_daily_pattern(predictions, 'results')
        plot_hourly_heatmap(predictions, 'results')
        plot_monthly_boxplot(predictions, 'results')
        plot_wind_speed_comparison(actual_wind, predictions, 'results')
        print('Wind power visualization plots saved.')

def plot_wind_speed_comparison(actual_wind, predicted_wind, save_dir):
    try:
        plt.figure(figsize=(12, 6))
        time_points = np.arange(len(predicted_wind))
        
        plt.plot(time_points, predicted_wind, 'orange', label='Predicted Wind', linewidth=1)
        if actual_wind is not None:
            plt.plot(time_points, actual_wind, 'b-', label='Actual Wind', linewidth=1)
        
        plt.title('Wind Speed Forecasting')
        plt.xlabel('Time')
        plt.ylabel('Normalized Wind Speed')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'wind_speed_comparison.png'))
        plt.close()
    except Exception as e:
        print(f'Error in plot_wind_speed_comparison: {e}')

if __name__ == '__main__':
    # Use the latest results directory
    results_dirs = [d for d in os.listdir() if d.startswith('results_')]
    if results_dirs:
        latest_results = max(results_dirs)
        create_visualizations(latest_results)
        print(f'Visualizations created using data from {latest_results}')
    else:
        print('No results directory found.')