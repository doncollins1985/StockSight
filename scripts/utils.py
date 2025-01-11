#!/usr/bin/env python
#================================
#  utils.py
#================================
"""
Utility functions and classes for loading config, validating files, scaling data,
building the model, callbacks, and custom metrics.
"""

import os
import json
import logging
import random
import time
import functools

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, List

from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GRU, Dense, Dropout,
    LayerNormalization, Bidirectional, MultiHeadAttention, Add,
    SpatialDropout1D, LeakyReLU, Permute, Concatenate, Flatten, Reshape
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import AdamW


##############################################################################
#                               CONFIG & LOGGING
##############################################################################
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and return the configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def validate_files(file_paths: List[str]) -> None:
    """
    Validates that each file in the list exists.
    Raises:
        TypeError if file_paths is not a list of strings/paths
        FileNotFoundError if any file does not exist
    """
    if not isinstance(file_paths, list):
        logging.error(f"validate_files expects a list, got {type(file_paths)}")
        raise TypeError("validate_files expects a list of file paths.")

    for path in file_paths:
        if not isinstance(path, (str, bytes, os.PathLike)):
            logging.error(f"Invalid path type: {path} (type: {type(path)})")
            raise TypeError(
                "Each file path should be a string, bytes, or os.PathLike."
            )
        if not os.path.exists(path):
            logging.error(f"File does not exist: {path}")
            raise FileNotFoundError(f"File does not exist: {path}")
        else:
            logging.info(f"Validated existence of file: {path}")


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


##############################################################################
#                              SCALING
##############################################################################
def scale_data(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> Tuple[np.ndarray, StandardScaler, np.ndarray, StandardScaler]:
    """
    Scale features and target using StandardScaler.

    Args:
        data (pd.DataFrame): The DataFrame containing all columns.
        feature_columns (list): Columns to be used as features.
        target_column (str): The column to be used as the target.

    Returns:
        features_scaled (np.ndarray): Scaled features array
        feature_scaler (StandardScaler): Fitted scaler for features
        target_scaled (np.ndarray): Scaled target array
        target_scaler (StandardScaler): Fitted scaler for target
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


def save_scalers(
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    scaler_X_filename: str,
    scaler_y_filename: str
) -> None:
    """
    Save the feature and target scalers to disk.

    Args:
        scaler_X (StandardScaler): Feature scaler
        scaler_y (StandardScaler): Target scaler
        scaler_X_filename (str): Where to save the feature scaler
        scaler_y_filename (str): Where to save the target scaler
    """
    os.makedirs(os.path.dirname(scaler_X_filename), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_y_filename), exist_ok=True)

    joblib.dump(scaler_X, scaler_X_filename)
    logging.info(f"Feature scaler saved to {scaler_X_filename}")

    joblib.dump(scaler_y, scaler_y_filename)
    logging.info(f"Target scaler saved to {scaler_y_filename}")


##############################################################################
#                             MODEL & METRICS
##############################################################################
def r2_keras(y_true, y_pred):
    """
    Custom R² metric for Keras.
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


def build_model(hp, input_shape: Tuple[int, int]) -> Model:
    """
    Build and compile a Keras model with hyperparameters from keras_tuner.

    Args:
        hp: HyperParameters object from keras_tuner
        input_shape: (timesteps, features)

    Returns:
        model (tf.keras.Model): Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Example hyperparameters for conv layers
    num_conv_layers = hp.Int('num_conv_layers', 2, 4, default=3)
    for i in range(num_conv_layers):
        filters = hp.Int(f'conv_filters_{i}', min_value=32, max_value=256, step=32, default=64)
        kernel_size = hp.Choice(f'conv_kernel_size_{i}', values=[2, 3, 5], default=3)
        x_conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(
                hp.Float(f'conv_l2_reg_{i}', 1e-6, 1e-3, sampling='log', default=1e-4)
            )
        )(x)
        x_conv = LeakyReLU(negative_slope=0.1)(x_conv)
        x_conv = LayerNormalization()(x_conv)
        drop_rate_conv = hp.Float(f'conv_dropout_rate_{i}', 0.1, 0.4, step=0.1, default=0.3)
        x_conv = SpatialDropout1D(rate=drop_rate_conv)(x_conv)

        if i > 0 and x_conv.shape[-1] == x.shape[-1]:
            # Residual
            x = Add()([x, x_conv])
        else:
            x = x_conv

        # Optional pooling
        x = MaxPooling1D(pool_size=2)(x)

    # Example hyperparameters for GRU layers
    num_gru_layers = hp.Int('num_gru_layers', 1, 2, default=2)
    for i in range(num_gru_layers):
        units = hp.Int(f'gru_units_{i}', min_value=64, max_value=256, step=64, default=128)
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
        gru_drop_rate = hp.Float(f'gru_dropout_rate_{i}', 0.2, 0.5, step=0.1, default=0.3)
        x_gru = Dropout(rate=gru_drop_rate)(x_gru)

        if i > 0 and x_gru.shape[-1] == x.shape[-1]:
            x = Add()([x, x_gru])
        else:
            x = x_gru

    # Multi-Head Attention
    num_heads_seq = hp.Int('attention_num_heads_seq', 8, 16, step=2, default=8)
    key_dim_seq = hp.Int('attention_key_dim_seq', 32, 128, step=32, default=64)
    x_attn_seq = MultiHeadAttention(num_heads=num_heads_seq, key_dim=key_dim_seq, dropout=0.1)(x, x)
    x = Add()([x, x_attn_seq])
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # Feature-wise attention
    x_perm = Permute((2, 1))(x)  # shape: (batch, features, timesteps)
    num_heads_feat = hp.Int('attention_num_heads_feat', 4, 8, step=2, default=4)
    key_dim_feat = hp.Int('attention_key_dim_feat', 16, 64, step=16, default=32)
    x_attn_feat = MultiHeadAttention(num_heads=num_heads_feat, key_dim=key_dim_feat, dropout=0.1)(x_perm, x_perm)
    x_feat = Add()([x_perm, x_attn_feat])
    x_feat = LayerNormalization()(x_feat)
    x_feat = Dropout(0.3)(x_feat)
    x_feat = Permute((2, 1))(x_feat)

    # Concatenate outputs from sequence-wise and feature-wise attention
    x = Concatenate()([x, x_feat])

    # Flatten
    x = Reshape((-1,))(x)

    # Dense layers
    num_dense_layers = hp.Int('num_dense_layers', 2, 8, default=4)
    for j in range(num_dense_layers):
        dense_units = hp.Int(f'dense_units_{j}', min_value=64, max_value=512, step=32, default=256)
        x_dense = Dense(
            units=dense_units,
            kernel_regularizer=regularizers.l2(
                hp.Float(f'dense_l2_reg_{j}', 1e-6, 1e-3, sampling='log', default=1e-4)
            )
        )(x)
        x_dense = LeakyReLU(negative_slope=0.1)(x_dense)
        x_dense = LayerNormalization()(x_dense)
        drop_rate_dense = hp.Float(f'dense_dropout_rate_{j}', 0.2, 0.5, step=0.1, default=0.3)
        x_dense = Dropout(rate=drop_rate_dense)(x_dense)
        x = x_dense

    outputs = Dense(1, activation='linear', dtype='float32')(x)

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


##############################################################################
#                           CUSTOM CALLBACKS
##############################################################################
class EpochTracker(tf.keras.callbacks.Callback):
    """
    Custom callback to track and save the current epoch number to a file.
    """
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'w') as f:
            f.write(str(epoch + 1))


class CosineAnnealingScheduler(Callback):
    """
    Cosine Annealing Learning Rate Scheduler with Warm Restarts.
    """
    def __init__(self, initial_lr: float, first_restart_epoch: int, T_mult=2):
        super().__init__()
        self.initial_lr = initial_lr
        self.first_restart_epoch = first_restart_epoch
        self.T_mult = T_mult
        self.current_epoch = 0
        self.T_cur = first_restart_epoch
        self.T_i = first_restart_epoch

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.compute_lr()
        logging.info(f"Epoch {epoch+1}: Computed learning rate: {lr}, type: {type(lr)}")
        logging.info(f"Optimizer learning rate before setting: {self.model.optimizer.learning_rate} ({type(self.model.optimizer.learning_rate)})")
        
        try:
            current_lr = self.model.optimizer.learning_rate.numpy()
        except AttributeError:
            current_lr = self.model.optimizer.learning_rate
        logging.info(f"Current optimizer learning rate value: {current_lr}")
        
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr)
            logging.debug(f"Epoch {epoch+1}: Setting learning rate to {lr:.6f}")
        else:
            try:
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                logging.debug(f"Epoch {epoch+1}: Setting learning rate to {lr:.6f}")
            except Exception as e:
                logging.error(f"Failed to set learning rate: {e}")
                raise

    def compute_lr(self):
        if self.current_epoch >= self.T_i:
            self.T_i *= self.T_mult
            self.T_cur = 0
        else:
            self.T_cur += 1

        cosine_decay = 0.5 * (1 + np.cos(np.pi * self.T_cur / self.T_i))
        lr = self.initial_lr * cosine_decay
        return lr

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1


##############################################################################
#                        CUSTOM COMBINED METRIC
##############################################################################
def combined_metric(
    y_true_rescaled: np.ndarray,
    y_pred_rescaled: np.ndarray,
    scaler: StandardScaler,
    weight_direction: float,
    weight_sharpe: float
) -> tf.Tensor:
    """
    Combined metric that balances directional accuracy and Sharpe ratio.

    Args:
        y_true_rescaled (np.ndarray): True target values (scaled back to original).
        y_pred_rescaled (np.ndarray): Predicted target values (scaled back).
        scaler (StandardScaler): The target scaler (unused here except for doc).
        weight_direction (float): Weight to apply to directional accuracy.
        weight_sharpe (float): Weight to apply to Sharpe ratio.

    Returns:
        A TF tensor with the combined metric value.
    """
    direction_true = tf.sign(y_true_rescaled[1:] - y_true_rescaled[:-1])
    direction_pred = tf.sign(y_pred_rescaled[1:] - y_pred_rescaled[:-1])

    direction_true = tf.cast(direction_true, tf.float32)
    direction_pred = tf.cast(direction_pred, tf.float32)

    directional_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(direction_true, direction_pred), tf.float32)
    ) * 100.0

    daily_returns = (y_pred_rescaled[1:] - y_pred_rescaled[:-1]) / y_pred_rescaled[:-1]
    # Avoid dividing by zero
    std_daily_returns = tf.math.reduce_std(daily_returns)
    sharpe_ratio = tf.cond(
        std_daily_returns > 0,
        lambda: (tf.reduce_mean(daily_returns) / std_daily_returns) * tf.sqrt(252.0),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

    combined = (weight_direction * directional_accuracy) + (weight_sharpe * sharpe_ratio)
    return combined


class CombinedMetricCallback(Callback):
    """
    Callback to compute a combined metric (direction + Sharpe) on validation data
    at the end of each epoch.
    """
    def __init__(
        self,
        scaler: StandardScaler,
        weight_direction: float,
        weight_sharpe: float
    ):
        super().__init__()
        self.scaler = scaler
        self.weight_direction = weight_direction
        self.weight_sharpe = weight_sharpe
        self.validation_data = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data is None:
            logging.warning("Validation data not set for CombinedMetricCallback.")
            return

        X_val, y_val = self.validation_data
        y_pred_all = self.model.predict(X_val, verbose=0)

        # If predictions are shape (samples, 1, 1), flatten them
        if y_pred_all.ndim == 3:
            y_pred_all = y_pred_all.reshape(-1, 1)

        # Inverse transform
        y_true_rescaled = self.scaler.inverse_transform(y_val.reshape(-1, 1))
        y_pred_rescaled = self.scaler.inverse_transform(y_pred_all.reshape(-1, 1))

        min_len = min(len(y_true_rescaled), len(y_pred_rescaled))
        y_true_rescaled = y_true_rescaled[:min_len]
        y_pred_rescaled = y_pred_rescaled[:min_len]

        val_combined = combined_metric(
            y_true_rescaled,
            y_pred_rescaled,
            self.scaler,
            self.weight_direction,
            self.weight_sharpe
        )
        logs["val_combined_metric"] = val_combined.numpy()
        self.model.history.history.setdefault("val_combined_metric", []).append(
            logs["val_combined_metric"]
        )

        logging.info(f"Epoch {epoch+1}: val_combined_metric: {logs['val_combined_metric']:.4f}")

