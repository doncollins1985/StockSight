#!/usr/bin/env python
#================================
#  data.py
#================================
"""
Script to:
1) Merge stock data with sentiment data.
2) Create scaled sequences for model training.
"""

import os
import logging
import numpy as np
import pandas as pd
import argparse
import joblib

from scripts.utils import (
    load_config, validate_files, log_time,
    scale_data, save_scalers
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data_setup.log"),
        logging.StreamHandler()
    ]
)


@log_time
def merge_sentiment_with_stock(input_file: str, sentiment_file: str, output_file: str) -> None:
    """
    Merge stock data with sentiment data on 'Date' and save the merged dataset.

    Args:
        input_file: Path to the stock data CSV
        sentiment_file: Path to the sentiment data CSV
        output_file: Path to save the merged dataset
    """
    logging.info(f"Merging '{input_file}' with sentiment '{sentiment_file}' -> '{output_file}'")
    validate_files([input_file, sentiment_file])

    # Load datasets
    stock_data = pd.read_csv(input_file, parse_dates=['Date'])
    logging.info(f"Loaded stock data from {input_file}")

    sentiment_data = pd.read_csv(sentiment_file, parse_dates=['Date'])
    logging.info(f"Loaded sentiment data from {sentiment_file}")

    # Merge on Date
    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='left')
    logging.info("Merged data on 'Date'")

    # Fill missing sentiment with neutral values
    fill_values = {
        'Positive': 0,
        'Negative': 0,
        'Neutral': 1,
        'Aggregate_Score': 0
    }
    merged_data.fillna(fill_values, inplace=True)
    logging.info("Filled missing sentiment data with default neutral values.")

    # Ensure output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    # Save merged dataset
    merged_data.to_csv(output_file, index=False)
    logging.info(f"Merged dataset saved to {output_file}")


@log_time
def create_sequences_with_sentiment(
    input_file: str,
    output_file: str,
    feature_columns: list,
    window_size: int,
    scaler_path_X: str,
    scaler_path_y: str
) -> None:
    """
    Create sequences (sliding windows) for time-series data (including sentiment),
    scale them, and save to a .npz file.

    Args:
        input_file: Path to merged stock+sentiment CSV
        output_file: Path to save the .npz with 'features' & 'labels'
        feature_columns: List of columns to include in each window
        window_size: Number of timesteps per sequence
        scaler_path_X: Where to save the fitted feature scaler
        scaler_path_y: Where to save the fitted target scaler
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} does not exist.")
        raise FileNotFoundError(f"Input file {input_file} not found.")

    data = pd.read_csv(input_file)
    logging.info(f"Loaded data for sequence creation from {input_file}")

    # Define the column we want to predict
    target_column_name = 'Close'
    if target_column_name not in data.columns:
        raise ValueError(f"Target column '{target_column_name}' not found in data.")

    # Check missing feature columns
    missing_cols = set(feature_columns) - set(data.columns)
    if missing_cols:
        logging.error(f"Missing columns in data: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Scale data
    features_scaled, feature_scaler, target_scaled, target_scaler = scale_data(
        data, feature_columns, target_column_name
    )

    # The index of 'Close' among the chosen feature columns
    target_idx = feature_columns.index(target_column_name)

    # Build sequences
    X = np.array([
        features_scaled[i : i + window_size]
        for i in range(len(features_scaled) - window_size)
    ])
    y = features_scaled[window_size:, target_idx]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save sequences
    np.savez(output_file, features=X, labels=y)
    logging.info(f"Sequences saved to {output_file}")

    # Save scalers
    os.makedirs(os.path.dirname(scaler_path_X), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path_y), exist_ok=True)
    save_scalers(feature_scaler, target_scaler, scaler_path_X, scaler_path_y)


def main(config_path: str) -> None:
    """
    Orchestrate data preparation:
    1) Merge sentiment with stock data
    2) Create scaled sequences
    """
    config = load_config(config_path)

    required_keys = [
        'stock_file', 'sentiment_file', 'merged_file',
        'sequence_file', 'feature_columns', 'window_size',
        'scaler_file_X', 'scaler_file_y'
    ]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        logging.error(f"Missing configuration keys: {missing_keys}")
        raise KeyError(f"Missing configuration keys: {missing_keys}")

    # Step 1: Merge data
    merge_sentiment_with_stock(
        input_file=config['stock_file'],
        sentiment_file=config['sentiment_file'],
        output_file=config['merged_file']
    )

    # Step 2: Create sequences
    feature_cols_extended = config['feature_columns'] + [
        'Positive', 'Negative', 'Neutral', 'Aggregate_Score'
    ]

    create_sequences_with_sentiment(
        input_file=config['merged_file'],
        output_file=config['sequence_file'],
        feature_columns=feature_cols_extended,
        window_size=config['window_size'],
        scaler_path_X=config['scaler_file_X'],
        scaler_path_y=config['scaler_file_y'],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data setup script.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="Path to the JSON configuration file."
    )
    args = parser.parse_args()
    main(args.config)

