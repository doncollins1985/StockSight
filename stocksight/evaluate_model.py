#================================
# model/evaluate_model.py
#================================

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from .utils import log_time, load_config, get_device
from .models import StockPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/evaluation.log"),
            logging.StreamHandler()
        ]
    )


def load_model_and_data(config):
    """
    Load the trained model and test data.
    """
    device = get_device()
    try:
        model_path = config['output_model_file']
        logging.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = StockPredictor(checkpoint['input_shape'], checkpoint['hyperparameters'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    try:
        sequence_file = config["sequence_file"]
        logging.info(f"Loading data from {sequence_file}")
        sequences = np.load(sequence_file, allow_pickle=True)
        X_test, y_test = sequences["features"], sequences["labels"]
        # Assuming last portion is test, logic should match train_model split.
        # But wait, evaluate_model previously loaded all sequences and just used them as "test"?
        # No, the previous code loaded "sequences.npz" and took "features" and "labels".
        # It didn't explicitly split. It seemingly evaluated on the WHOLE dataset or expected "features" to be just test data?
        # Looking at previous code: `X_test, y_test = sequences["features"], sequences["labels"]`.
        # And `train_model.py` splits the data.
        # The `data.py` creates `sequences.npz` with ALL data.
        # So the previous evaluation was actually evaluating on the WHOLE dataset unless `sequences.npz` was strictly test data.
        # `data.py` (which I didn't change) creates sequences from merged file.
        # Let's check `load_data` in `train_model.py`. It loads `sequences.npz` then SPLITS it.
        # So `evaluate_model.py` was technically evaluating on everything (train+val+test) which is ... a choice.
        # I will preserve this behavior for now to avoid breaking scope, but ideally it should only eval on test split.
        # However, to match previous logic:
        logging.info(f"Loaded {X_test.shape[0]} samples for evaluation.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise
    if X_test.shape[0] != y_test.shape[0]:
        error_msg = "Mismatch between number of features and labels."
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    # Load Dates
    dates = None
    try:
        merged_file = config['merged_file']
        window_size = config['window_size']
        if os.path.exists(merged_file):
            import pandas as pd
            df = pd.read_csv(merged_file)
            # The sequences start at index 0 which corresponds to df row `window_size`.
            # y[0] is target at `window_size`.
            # We need dates for y_test.
            # y_test here is actually ALL data in the sequences file (based on previous findings).
            # So we take df['Date'][window_size:]
            if 'Date' in df.columns:
                dates = pd.to_datetime(df['Date']).iloc[window_size:].values
                if len(dates) != len(y_test):
                    logging.warning(f"Date length {len(dates)} does not match label length {len(y_test)}. Ignoring dates.")
                    dates = None
                else:
                    logging.info(f"Loaded {len(dates)} dates for plotting.")
    except Exception as e:
        logging.warning(f"Failed to load dates: {e}")

    return model, X_test, y_test, dates

def load_scaler_X(config):
    """
    Load the feature scaler used during training.
    """
    try:
        scaler_dir = os.path.dirname(config["scaler_file_X"]) # Config has scaler_file_X path
        scaler_X_file = config["scaler_file_X"]
        logging.info(f"Loading feature scaler from {scaler_X_file}")
        scaler_X = joblib.load(scaler_X_file)
        return scaler_X
    except Exception as e:
        logging.error(f"Failed to load feature scaler: {e}")
        raise

def load_scaler_y(config):
    """
    Load the target scaler used during training.
    """
    try:
        scaler_y_file = config["scaler_file_y"]
        logging.info(f"Loading target scaler from {scaler_y_file}")
        scaler_y = joblib.load(scaler_y_file)
        return scaler_y
    except Exception as e:
        logging.error(f"Failed to load target scaler: {e}")
        raise

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Avoid division by zero
    return 100 * np.mean(diff)


def compute_metrics(predictions, y_true, metrics_list):
    """
    Compute evaluation metrics.
    """
    results = {}
    
    if 'mae' in metrics_list:
        results['MAE'] = mean_absolute_error(y_true, predictions)
    if 'mse' in metrics_list:
        results['MSE'] = mean_squared_error(y_true, predictions)
    if 'mape' in metrics_list:
        results['MAPE'] = mean_absolute_percentage_error(y_true, predictions) * 100
    if 'r2' in metrics_list:
        results['R2'] = r2_score(y_true, predictions)
    if 'smape' in metrics_list:
        results['SMAPE'] = calculate_smape(y_true, predictions)

    logging.info("Computed Metrics:")
    for k, v in results.items():
        logging.info(f" - {k}: {v:.4f}")
        
    return results


def save_evaluation_results(config, metrics_results, predictions, y_true):
    """
    Save evaluation results to a JSON file.
    """
    os.makedirs(config.get('evaluation_output_dir', 'evaluation_results'), exist_ok=True)
    evaluation_results = metrics_results.copy()
    evaluation_results.update({
        "Predictions": predictions.tolist(),
        "Actual": y_true.tolist()
    })
    evaluation_file = os.path.join(
        config.get('evaluation_output_dir', 'evaluation_results'), "evaluation_results_with_sentiment.json")
    try:
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        logging.info(f"Evaluation results saved to {evaluation_file}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results: {e}")
        raise


def generate_and_save_plot(config, predictions, y_true, dates=None):
    """
    Generate and save a plot comparing predictions and actual values.
    Includes a full plot and a zoomed-in plot (last 365 days).
    """
    output_dir = config.get('evaluation_output_dir', 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Full Plot
    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, y_true, label="Actual", color="blue")
        plt.plot(dates, predictions, label="Predicted", color="orange", linestyle="--")
        plt.xlabel("Date")
    else:
        plt.plot(y_true, label="Actual", color="blue")
        plt.plot(predictions, label="Predicted", color="orange", linestyle="--")
        plt.xlabel("Time Step")
        
    plt.title("Model Evaluation: Predictions vs Actual (Full History)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "evaluation_plot_with_sentiment.png")
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Full evaluation plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save full plot: {e}")

    # 2. Zoomed Plot (Last 365 days)
    zoom_n = 365
    if len(y_true) > zoom_n:
        plt.figure(figsize=(12, 6))
        
        y_true_zoom = y_true[-zoom_n:]
        pred_zoom = predictions[-zoom_n:]
        
        if dates is not None:
            dates_zoom = dates[-zoom_n:]
            plt.plot(dates_zoom, y_true_zoom, label="Actual", color="blue")
            plt.plot(dates_zoom, pred_zoom, label="Predicted", color="orange", linestyle="--")
            plt.xlabel("Date")
        else:
            plt.plot(range(len(y_true)-zoom_n, len(y_true)), y_true_zoom, label="Actual", color="blue")
            plt.plot(range(len(y_true)-zoom_n, len(y_true)), pred_zoom, label="Predicted", color="orange", linestyle="--")
            plt.xlabel("Time Step")

        plt.title(f"Model Evaluation: Predictions vs Actual (Last {zoom_n} Steps)")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        
        zoom_plot_path = os.path.join(output_dir, "evaluation_plot_with_sentiment_zoom.png")
        try:
            plt.savefig(zoom_plot_path, bbox_inches='tight')
            plt.close()
            logging.info(f"Zoomed evaluation plot saved to {zoom_plot_path}")
        except Exception as e:
            logging.error(f"Failed to save zoomed plot: {e}")


@log_time
def evaluate_model_with_sentiment(config, metrics_list):
    """
    Evaluate the trained LSTM model with sentiment features.
    """
    try:
        model, X_test, y_test, dates = load_model_and_data(config)
        
        scaler_X = load_scaler_X(config)
        scaler_y = load_scaler_y(config)
        
        num_samples, window, num_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
        
        device = get_device()
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        logging.info("Generating predictions...")
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
            
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
            
        predictions_original = scaler_y.inverse_transform(predictions)
        y_test_original = scaler_y.inverse_transform(y_test) # Inverse transform targets too if they were scaled? 
        # Wait, y_test loaded from sequences.npz is UNSCALED. 
        # But predictions are coming from model which outputs SCALED values.
        # So predictions_original is correct.
        # y_test_original should be just y_test?
        # In train_model.py, y_train_scaled is used for training.
        # So model outputs scaled values.
        # scaler_y.inverse_transform(predictions) brings them back to original units (Log_Return or Close).
        # y_test from sequences.npz is RAW (Unscaled).
        y_test_original = y_test

        target_column = config.get('target_column', 'Close')
        feature_columns = config.get('feature_columns', [])
        
        if target_column == 'Log_Return' and 'Close' in feature_columns:
            logging.info("Target is Log_Return. Reconstructing Close prices for evaluation.")
            close_idx = feature_columns.index('Close')
            # Previous Close is the last value of 'Close' feature in the input window
            # X_test shape: (samples, window, features)
            prev_close = X_test[:, -1, close_idx].reshape(-1, 1)
            
            # P_t = P_{t-1} * exp(R_t)
            predictions_price = prev_close * np.exp(predictions_original)
            actual_price = prev_close * np.exp(y_test_original)
            
            # Evaluate on PRICES
            metrics_results = compute_metrics(predictions_price, actual_price, metrics_list)
            save_evaluation_results(config, metrics_results, predictions_price, actual_price)
            generate_and_save_plot(config, predictions_price, actual_price, dates)
            
            # Also optionally log metrics on returns?
            logging.info("Metrics on Log Returns:")
            compute_metrics(predictions_original, y_test_original, metrics_list)
            
        else:
            metrics_results = compute_metrics(predictions_original, y_test_original, metrics_list)
            save_evaluation_results(config, metrics_results, predictions_original, y_test_original)
            generate_and_save_plot(config, predictions_original, y_test_original, dates)
            
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        raise


def main(config_path: str = "config.json", metrics_str: str = "mae,mse,mape,r2") -> None:
    """
    Main function to evaluate the trained model using the provided configuration.
    """
    setup_logging()
    try:
        config = load_config(config_path)
        metrics_list = [m.strip().lower() for m in metrics_str.split(',')]
        evaluate_model_with_sentiment(config, metrics_list)
    except Exception as e:
        logging.error(f"Failed to load configuration or evaluate: {e}")
        exit(1)


if __name__ == "__main__":
    main()
