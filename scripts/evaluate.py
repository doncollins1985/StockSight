#!/usr/bin/env python
#================================
#  evaluate.py
#================================
"""
Script to evaluate a trained model on the test set, replicate feature engineering,
load scalers, compute metrics, and show plots of predictions vs actual.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import tensorflow as tf
import joblib
import pandas as pd
from typing import Dict

from scripts.utils import (
    load_config, validate_files,
    scale_data, save_scalers, log_time,
    r2_keras, combined_metric
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log")
    ]
)


@log_time
def evaluate_model(config_path: str) -> Dict[str, float]:
    """
    Evaluate a trained model on the test set and plot predicted vs. actual prices.
    Replicates the same feature engineering and scaling steps used during training.

    Args:
        config_path (str): Path to the JSON config

    Returns:
        Dict[str, float]: Various metrics from evaluation
    """
    # 1) Load config
    config = load_config(config_path)
    required_keys = [
        "input_file", "sequence_file", "feature_columns", "target_column",
        "lookahead_period", "train_ratio", "val_ratio",
        "output_model_file", "scaler_dir", "feature_engineering_window"
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing configuration parameters: {missing_keys}")

    # 2) Validate existence
    model_file = config["output_model_file"]
    input_file = config["input_file"]
    sequence_file = config["sequence_file"]
    validate_files([model_file, input_file, sequence_file])

    scaler_X_filename = os.path.join(config["scaler_dir"], "scaler_X.save")
    scaler_y_filename = os.path.join(config["scaler_dir"], "scaler_y.save")
    validate_files([scaler_X_filename, scaler_y_filename])

    # 3) Load scalers
    scaler_X = joblib.load(scaler_X_filename)
    scaler_y = joblib.load(scaler_y_filename)

    # 4) Load original data
    data = pd.read_csv(input_file)
    logging.info(f"Data loaded from {input_file}, shape: {data.shape}")

    # 5) Replicate feature engineering
    window_size = config["feature_engineering_window"]
    feature_cols = config["feature_columns"]
    for col in feature_cols:
        data[f'{col}_rolling_mean_{window_size}'] = data[col].rolling(window=window_size).mean()
        data[f'{col}_rolling_std_{window_size}'] = data[col].rolling(window=window_size).std()
    data.dropna(inplace=True)

    # 6) Define columns used in training
    all_features = feature_cols + \
        [f'{col}_rolling_mean_{window_size}' for col in feature_cols] + \
        [f'{col}_rolling_std_{window_size}' for col in feature_cols]

    # 7) Load preprocessed sequences
    sequences_data = np.load(sequence_file, allow_pickle=True)
    X_all = sequences_data['features']

    # 8) Lookahead shift
    lookahead_period = config.get("lookahead_period", 7)
    target_values = data[config["target_column"]].values
    X_all = X_all[:-lookahead_period]
    y_all = target_values[lookahead_period:]

    # 9) Split out test portion
    total_samples = len(X_all)
    train_split = int(config["train_ratio"] * total_samples)
    val_split = int(config["val_ratio"] * total_samples)
    test_split = total_samples - train_split - val_split

    X_test = X_all[train_split + val_split:]
    y_test_unscaled = y_all[train_split + val_split:]

    # 10) Scale X_test
    num_features = X_test.shape[-1]
    X_test_reshaped = X_test.reshape(-1, num_features)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

    # 11) Scale y_test for direct comparison
    y_test_scaled = scaler_y.transform(y_test_unscaled.reshape(-1, 1)).flatten()

    # 12) Load model
    model = tf.keras.models.load_model(model_file, custom_objects={'r2_keras': r2_keras})
    logging.info(f"Loaded model from {model_file}")

    # 13) Predict
    predictions_scaled = model.predict(X_test_scaled)
    if predictions_scaled.ndim == 3:
        predictions_scaled = predictions_scaled.reshape(-1, 1)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Ensure shape match
    min_len = min(len(predictions), len(y_test_unscaled))
    predictions = predictions[:min_len].flatten()
    y_test = y_test_unscaled[:min_len]

    # 14) Metrics
    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)

    direction_true = np.sign(y_test[1:] - y_test[:-1])
    direction_pred = np.sign(predictions[1:] - predictions[:-1])
    directional_accuracy = np.mean(direction_true == direction_pred) * 100

    daily_returns = (predictions[1:] - predictions[:-1]) / predictions[:-1]
    sharpe_ratio = 0.0
    if np.std(daily_returns) != 0:
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    cumulative_returns = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    weight_direction = config.get("weight_direction", 0.5)
    weight_sharpe = config.get("weight_sharpe", 0.5)

    # For the combined metric, we re-scale y_test and preds
    y_test_scaled_again = scaler_y.transform(y_test.reshape(-1, 1))
    preds_scaled_again  = scaler_y.transform(predictions.reshape(-1, 1))

    test_combined_metric = combined_metric(
        y_test_scaled_again,
        preds_scaled_again,
        scaler_y,
        weight_direction,
        weight_sharpe
    ).numpy()

    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAPE: {mape:.4f}")
    logging.info(f"R2: {r2:.4f}")
    logging.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    logging.info(f"Max Drawdown: {max_drawdown:.4f}")
    logging.info(f"Combined Metric: {test_combined_metric:.4f}")

    # 15) Plot Actual vs Predicted
    plt.figure(figsize=(14, 6))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # 16) Residuals
    residuals = y_test - predictions
    plt.figure(figsize=(14, 6))
    plt.scatter(predictions, residuals, alpha=0.5, c='green')
    plt.hlines(y=0, xmin=min(predictions), xmax=max(predictions), colors='r')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.show()

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Directional Accuracy": directional_accuracy,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Combined Metric": test_combined_metric
    }
    logging.info(f"Evaluation Results: {results}")
    return results


def main():
    # You could optionally parse argparse here if desired:
    # e.g. python evaluate.py --config config.json
    results = evaluate_model("config.json")
    logging.info(f"Evaluation Results: {results}")


if __name__ == "__main__":
    main()

