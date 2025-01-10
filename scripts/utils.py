import os
import json
import logging
import random
import time
import functools

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, GRU, Dense, Dropout,
    LayerNormalization, Bidirectional, MultiHeadAttention, Add,
    SpatialDropout1D, LeakyReLU, Permute, Concatenate, Flatten, 				Reshape
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import AdamW

import joblib

from keras_tuner import HyperParameters

# Example custom metric for R^2 (not necessarily exact)
def r2_keras(y_true, y_pred):
    residual = y_true - y_pred
    total = y_true - np.mean(y_true)
    return 1 - (np.sum(residual**2) / np.sum(total**2))
  
# --- Helper Functions ---
def load_config(config_path):
    """
    Load and return the configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class EpochTracker(tf.keras.callbacks.Callback):
    """
    Custom callback to track and save the current epoch number.
    """
    def __init__(self, filepath):
        super(EpochTracker, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'w') as f:
            f.write(str(epoch + 1))

class CosineAnnealingScheduler(Callback):
    """
    Cosine Annealing Learning Rate Scheduler with Warm Restarts.
    """
    def __init__(self, initial_lr, first_restart_epoch, T_mult=2):
        super(CosineAnnealingScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.first_restart_epoch = first_restart_epoch
        self.T_mult = T_mult
        self.current_epoch = 0
        self.T_cur = first_restart_epoch
        self.T_i = first_restart_epoch

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.compute_lr()
        # Reduced logging verbosity
        logging.debug(f"Epoch {epoch+1}: Setting learning rate to {lr:.6f}")

    def compute_lr(self):
        if self.current_epoch >= self.T_i:
            self.T_i = self.T_i * self.T_mult
            self.T_cur = 0
        else:
            self.T_cur += 1
        cosine_decay = 0.5 * (1 + np.cos(np.pi * self.T_cur / self.T_i))
        lr = self.initial_lr * cosine_decay
        return lr

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1

def r2_keras(y_true, y_pred):
    """
    Custom R² metric for Keras.
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

def scale_data(data: pd.DataFrame, feature_columns: list, target_column: str):
    """
    Scale features and target using StandardScaler.
    """
    try:
        feature_scaler = StandardScaler()
        features = data[feature_columns].values
        features_scaled = feature_scaler.fit_transform(features)
        logging.info("Features scaled using StandardScaler.")

        target_scaler = StandardScaler()
        target = data[target_column].values.reshape(-1, 1)
        target_scaled = target_scaler.fit_transform(target).flatten()
        logging.info("Target scaled using StandardScaler.")

        return features_scaled, feature_scaler, target_scaled, target_scaler
    except Exception as e:
        logging.exception("An error occurred during scaling.")
        raise e

def save_scalers(scaler_X, scaler_y, scaler_X_filename: str, scaler_y_filename: str):
    """
    Save the feature and target scalers to disk.
    """
    os.makedirs(os.path.dirname(scaler_X_filename), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_y_filename), exist_ok=True)

    joblib.dump(scaler_X, scaler_X_filename)
    logging.info(f"Feature scaler saved to {scaler_X_filename}")

    joblib.dump(scaler_y, scaler_y_filename)
    logging.info(f"Target scaler saved to {scaler_y_filename}")


def log_time(func):
    """
    Decorator to log the execution time of functions.
    """
    @functools.wraps(func)
    def wrapper_log_time(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Started '{func.__name__}'...")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Finished '{func.__name__}' in {elapsed_time:.2f} seconds.")
        return result

    return wrapper_log_time

def validate_files(file_paths):
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

def load_data(input_file, sequence_file, feature_columns, target_column, config):
    """
    Load and prepare the data with a lookahead period for the target variable.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not os.path.exists(sequence_file):
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

    data = pd.read_csv(input_file)
    logging.info(f"Data loaded from {input_file}, shape: {data.shape}")

    # Feature Engineering (Rolling mean and std)
    window_size = config.get("feature_engineering_window", 5)
    for col in feature_columns:
        data[f'{col}_rolling_mean_{window_size}'] = data[col].rolling(window=window_size).mean()
        data[f'{col}_rolling_std_{window_size}'] = data[col].rolling(window=window_size).std()
    data.dropna(inplace=True)
    logging.info(f"Feature engineering applied with window size {window_size}.")
    
    # Scale data
    features_scaled, feature_scaler, target_scaled, target_scaler = scale_data(
        data, 
        feature_columns + [f'{col}_rolling_mean_{window_size}' for col in feature_columns] +
            [f'{col}_rolling_std_{window_size}' for col in feature_columns],
        target_column
    )

    # Create a new DataFrame with scaled data
    scaled_data = pd.DataFrame(features_scaled, columns=feature_columns + [f'{col}_rolling_mean_{window_size}' for col in feature_columns] +
            [f'{col}_rolling_std_{window_size}' for col in feature_columns])
    scaled_data[target_column] = target_scaled

    # Save scaled data to CSV
    scaled_data_file = input_file.replace('.csv', '_scaled.csv')
    scaled_data.to_csv(scaled_data_file, index=False)
    logging.info(f"Scaled data saved to {scaled_data_file}")

    # Load preprocessed sequences
    sequences_data = np.load(sequence_file, allow_pickle=True)
    X = sequences_data['features']

    # Apply lookahead period to the target variable
    lookahead_period = config.get("lookahead_period", 7)
    y = target_scaled[lookahead_period:]
    X = X[:-lookahead_period]

    logging.info(
        f"Sequences loaded from {sequence_file} with {lookahead_period}-day lookahead. "
        f"X shape: {X.shape}, y shape: {y.shape}"
    )

    # Data splitting
    total_samples = len(X)
    train_split = int(config["train_ratio"] * total_samples)
    val_split = int(config["val_ratio"] * total_samples)
    test_split = total_samples - train_split - val_split

    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:train_split + val_split]
    y_val = y[train_split:train_split + val_split]
    X_test = X[train_split + val_split:]
    y_test = y[train_split + val_split:]
    logging.info("Data splitting completed correctly.")

    # Feature scaling using StandardScaler
    scaler_X = StandardScaler()
    num_samples, window, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_train_scaled = scaler_X.fit_transform(
        X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(
        X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(
        X_test.reshape(-1, num_features)).reshape(X_test.shape)
    logging.info("Feature scaling applied using StandardScaler.")

    # Remove any extra singleton dimensions
    if X_train_scaled.ndim == 4 and X_train_scaled.shape[-1] == 1:
        X_train_scaled = X_train_scaled.squeeze(-1)
        X_val_scaled = X_val_scaled.squeeze(-1)
        X_test_scaled = X_test_scaled.squeeze(-1)
        logging.info("Removed extra singleton dimension from X_train_scaled.")
    else:
        logging.info(f"X_train_scaled has shape: {X_train_scaled.shape}")

    # Target scaling using StandardScaler
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    logging.info("Target scaling applied using StandardScaler.")

    logging.info(f"Training samples: {len(X_train_scaled)}")
    logging.info(f"Validation samples: {len(X_val_scaled)}")
    logging.info(f"Testing samples: {len(X_test_scaled)}")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler


def build_model(hp, input_shape):
    """
    Build and compile a redesigned Keras model, accommodating both technical 
    and sentiment-derived features. 
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # TCN-like Blocks for Initial Feature Extraction
    num_tcn_blocks = hp.Int('num_tcn_blocks', 1, 3, default=2)
    for i in range(num_tcn_blocks):
        filters = hp.Int(
            f'tcn_filters_{i}',
            min_value=32,
            max_value=256,
            step=32,
            default=64
        )
        kernel_size = hp.Choice(
            f'tcn_kernel_size_{i}',
            values=[2, 3, 5],
            default=3
        )
        x_tcn = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** i,
            padding='causal',
            kernel_regularizer=regularizers.l2(
                hp.Float(f'tcn_l2_reg_{i}', 1e-6, 1e-3, sampling='log', default=1e-4)
            )
        )(x)
        x_tcn = LeakyReLU(negative_slope=0.1)(x_tcn)
        x_tcn = LayerNormalization()(x_tcn)
        x_tcn = SpatialDropout1D(
            rate=hp.Float(f'tcn_dropout_rate_{i}', 0.1, 0.4, step=0.1, default=0.3)
        )(x_tcn)

        if x_tcn.shape[-1] == x.shape[-1]:
            x = Add()([x, x_tcn])
        else:
            x = x_tcn

    # GRU Layers
    num_gru_layers = hp.Int('num_gru_layers', 1, 2, default=1)
    for i in range(num_gru_layers):
        units = hp.Int(f'gru_units_{i}', 64, 256, step=64, default=128)
        x_gru = Bidirectional(
            GRU(
                units=units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(
                    hp.Float(f'gru_l2_reg_{i}', 1e-6, 1e-3, sampling='log', default=1e-4)
                )
            )
        )(x)
        x_gru = LeakyReLU(negative_slope=0.1)(x_gru)
        x_gru = LayerNormalization()(x_gru)
        x_gru = Dropout(
            rate=hp.Float(f'gru_dropout_rate_{i}', 0.2, 0.5, step=0.1, default=0.3)
        )(x_gru)

        if x_gru.shape[-1] == x.shape[-1]:
            x = Add()([x, x_gru])
        else:
            x = x_gru

    # Single Multi-Head Attention (Temporal Focus)
    num_heads = hp.Int('attention_num_heads', 2, 8, step=2, default=4)
    key_dim = hp.Int('attention_key_dim', 16, 64, step=16, default=32)
    x_attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.1
    )(x, x)
    x = Add()([x, x_attn])
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # Flatten for Dense Layers
    x = Flatten()(x)

    # Dense Layers
    num_dense_layers = hp.Int('num_dense_layers', 1, 4, default=2)
    for j in range(num_dense_layers):
        units = hp.Int(
            f'dense_units_{j}',
            min_value=64,
            max_value=512,
            step=64,
            default=128
        )
        x = Dense(
            units=units,
            kernel_regularizer=regularizers.l2(
                hp.Float(f'dense_l2_reg_{j}', 1e-6, 1e-3, sampling='log', default=1e-4)
            )
        )(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = LayerNormalization()(x)
        x = Dropout(
            rate=hp.Float(f'dense_dropout_rate_{j}', 0.1, 0.5, step=0.1, default=0.3)
        )(x)

    # Output Layer
    outputs = Dense(1, activation='linear', dtype='float32')(x)

    # Optimizer and Compilation
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    optimizer = AdamW(
        learning_rate=learning_rate,
        weight_decay=hp.Float('weight_decay', 1e-5, 1e-3, sampling='log', default=1e-4),
        clipnorm=1.0
    )

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae', 'mape', r2_keras]
    )

    return model


def combined_metric(y_true_rescaled, y_pred_rescaled, scaler, weight_direction, weight_sharpe):
    """
    Combined metric for evaluating model performance, balancing directional accuracy and Sharpe ratio.
    """
    # Calculate Directional Accuracy
    print("y_true_rescaled shape in combined_metric:", y_true_rescaled.shape)
    print("y_pred_rescaled shape in combined_metric:", y_pred_rescaled.shape)

    direction_true = tf.sign(y_true_rescaled[1:] - y_true_rescaled[:-1])
    direction_pred = tf.sign(y_pred_rescaled[1:] - y_pred_rescaled[:-1])

    print("direction_true shape:", direction_true.shape)
    print("direction_pred shape:", direction_pred.shape)

    # Explicitly cast to tf.float32
    direction_true = tf.cast(direction_true, tf.float32)
    direction_pred = tf.cast(direction_pred, tf.float32)

    directional_accuracy = tf.reduce_mean(tf.cast(tf.equal(direction_true, direction_pred), tf.float32)) * 100

    # Calculate Sharpe Ratio (annualized, assuming risk-free rate of 0)
    daily_returns = (y_pred_rescaled[1:] - y_pred_rescaled[:-1]) / y_pred_rescaled[:-1]
    sharpe_ratio = tf.reduce_mean(daily_returns) / tf.math.reduce_std(daily_returns) * tf.sqrt(tf.constant(252, dtype=tf.float32))

    # Combine the metrics with weights
    combined = (weight_direction * directional_accuracy) + (weight_sharpe * sharpe_ratio)
    return combined


class CombinedMetricCallback(Callback):
    def __init__(self, scaler, weight_direction, weight_sharpe):
        super(CombinedMetricCallback, self).__init__()
        self.scaler = scaler
        self.weight_direction = weight_direction
        self.weight_sharpe = weight_sharpe
        self.validation_data = None

    # def on_epoch_begin(self, epoch, logs=None):
    #     logs = logs or {}
    #     logs['val_combined_metric'] = 0 

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data is not None:
            y_true = self.validation_data[1]
            X_val = self.validation_data[0]

            # Get predictions for the entire validation set
            y_pred_all = self.model.predict(X_val, verbose=0)

            # Reshape y_pred_all if needed
            if y_pred_all.ndim == 3:
                y_pred_all = y_pred_all.reshape(-1, 1)

            # Rescale y_true and y_pred_all
            y_true_rescaled = self.scaler.inverse_transform(y_true.reshape(-1, 1))
            y_pred_rescaled = self.scaler.inverse_transform(y_pred_all.reshape(-1, 1))

            # Ensure y_true and y_pred have the same length before slicing
            min_len = min(len(y_true_rescaled), len(y_pred_rescaled))
            y_true_rescaled = y_true_rescaled[:min_len]
            y_pred_rescaled = y_pred_rescaled[:min_len]

            print("y_true_rescaled shape:", y_true_rescaled.shape)
            print("y_pred_rescaled shape:", y_pred_rescaled.shape)

            # Calculate metrics using the entire validation set
            val_combined = combined_metric(
                y_true_rescaled,
                y_pred_rescaled,
                self.scaler,
                self.weight_direction,
                self.weight_sharpe,
            )
            logs["val_combined_metric"] = val_combined.numpy()
            self.model.history.history.setdefault(
                "val_combined_metric", []
            ).append(logs["val_combined_metric"])
            logging.info(
                f"Epoch {epoch+1}: val_combined_metric: {logs['val_combined_metric']:.4f}"
            )
        else:
            logging.warning(
                "Validation data not set for CombinedMetricCallback."
            )
