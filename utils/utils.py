#================================
# utils/utils.py
#================================

import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')


def r2_keras(y_true, y_pred):
    """
    Custom R² metric for Keras.
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


def validate_files(file_paths: list) -> None:
    """
    Validates that each file in the list exists.
    """
    if not isinstance(file_paths, list):
        logging.error(f"validate_files expects a list, got {type(file_paths)} instead.")
        raise TypeError("validate_files expects a list of file paths.")
    for path in file_paths:
        if not isinstance(path, (str, bytes, os.PathLike)):
            logging.error(f"Invalid path type: {path} (type: {type(path)})")
            raise TypeError(f"Each file path should be a string, bytes, or os.PathLike, got {type(path)} instead.")
        if not os.path.exists(path):
            logging.error(f"File does not exist: {path}")
            raise FileNotFoundError(f"File does not exist: {path}")
        else:
            logging.info(f"Validated existence of file: {path}")


def load_config(config_path: str) -> dict:
    """
    Loads configuration from a JSON file.
    """
    if not isinstance(config_path, (str, bytes, os.PathLike)):
        logging.error(f"Config path must be a string, bytes, or os.PathLike, got {type(config_path)} instead.")
        raise TypeError("Config path must be a string, bytes, or os.PathLike.")
    if not os.path.exists(config_path):
        logging.error(f"Config file does not exist: {config_path}")
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config


def log_time(func):
    """
    Decorator to log the execution time of a function.
    """
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Started '{func.__name__}'")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Finished '{func.__name__}' in {elapsed_time:.2f} seconds")
    return wrapper


def scale_data(data: pd.DataFrame, feature_columns: list, target_column: str, scaler_type: str = 'minmax') -> tuple:
    """
    Scale features and target values.
    """
    if scaler_type.lower() == 'minmax':
        scaler_class = MinMaxScaler
    elif scaler_type.lower() == 'standard':
        scaler_class = StandardScaler
    else:
        logging.error(f"Unsupported scaler_type: {scaler_type}")
        raise ValueError("scaler_type must be 'minmax' or 'standard'.")
    features = data[feature_columns].values
    feature_scaler = scaler_class()
    target_scaler = scaler_class()
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(data[[target_column]])
    logging.info("Data scaling completed using {} scaler.".format(scaler_type))
    return features_scaled, feature_scaler, target_scaled, target_scaler


def save_scalers(feature_scaler: object, target_scaler: object, scaler_file_X: str, scaler_file_y: str) -> None:
    """
    Save scalers to files.
    """
    try:
        os.makedirs(os.path.dirname(scaler_file_X), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_file_y), exist_ok=True)
        with open(scaler_file_X, 'wb') as f:
            pickle.dump(feature_scaler, f)
        with open(scaler_file_y, 'wb') as f:
            pickle.dump(target_scaler, f)
        logging.info(f"Feature scaler saved to {scaler_file_X}")
        logging.info(f"Target scaler saved to {scaler_file_y}")
    except Exception as e:
        logging.error(f"Failed to save scalers: {e}")
        raise


def load_scalers(scaler_file_X: str, scaler_file_y: str) -> tuple:
    """
    Load feature and target scalers from files.
    """
    try:
        with open(scaler_file_X, 'rb') as f:
            feature_scaler = pickle.load(f)
        with open(scaler_file_y, 'rb') as f:
            target_scaler = pickle.load(f)
        logging.info("Scalers loaded successfully.")
        return feature_scaler, target_scaler
    except Exception as e:
        logging.error(f"Failed to load scalers: {e}")
        raise


def get_lr_scheduler(initial_lr: float, decay_rate: float, decay_steps: int):
    """
    Returns a Keras LearningRateScheduler callback that decays the learning rate.
    """
    def scheduler(epoch, lr):
        if epoch > 0 and epoch % decay_steps == 0:
            new_lr = lr * decay_rate
            logging.info(f"Learning rate decayed from {lr:.6f} to {new_lr:.6f} at epoch {epoch}")
            return new_lr
        return lr
    return tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)


def plot_predictions(dates, actual, predicted, title="Stock Price Prediction", save_path=None):
    """
    Plot actual vs predicted stock prices.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()

