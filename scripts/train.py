#!/usr/bin/env python
#======================================
#  train.py
#======================================
"""
Script to train a time-series model with optional hyperparameter tuning.
Saves best model, scalers, and logs.
"""

import os
import json
import logging
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import AdamW

import keras_tuner as kt
import matplotlib.pyplot as plt

from scripts.utils import (
    EpochTracker, CosineAnnealingScheduler, CombinedMetricCallback, r2_keras,
    load_config, validate_files, build_model, combined_metric, scale_data,
    save_scalers, log_time
)


##############################################################################
#                          GLOBAL CONFIG & LOGGING
##############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)


##############################################################################
#                             HELPER FUNCTIONS
##############################################################################
def load_data(input_file: str, sequence_file: str, feature_columns: list,
              target_column: str, config: dict):
    """
    Load CSV data, apply rolling features, scale them, then load the .npz
    sequences to match. Split into train/val/test sets.

    The code then does a second scaling pass on the sequences themselves,
    which may or may not be what you want. Adjust if needed.
    """
    validate_files([input_file, sequence_file])

    data = pd.read_csv(input_file)
    logging.info(f"Data loaded from {input_file}, shape: {data.shape}")

    # Rolling
    window_size = config.get("feature_engineering_window", 5)
    for col in feature_columns:
        data[f'{col}_rolling_mean_{window_size}'] = data[col].rolling(window=window_size).mean()
        data[f'{col}_rolling_std_{window_size}'] = data[col].rolling(window=window_size).std()

    data.dropna(inplace=True)
    logging.info(f"Feature engineering (rolling mean/std) with window size {window_size} done.")

    # Scale with new columns
    all_features = feature_columns + [
        f'{col}_rolling_mean_{window_size}' for col in feature_columns
    ] + [
        f'{col}_rolling_std_{window_size}' for col in feature_columns
    ]

    features_scaled, feature_scaler, target_scaled, target_scaler = scale_data(
        data, all_features, target_column
    )

    # Save a scaled version of the CSV
    scaled_data_file = input_file.replace('.csv', '_scaled.csv')
    scaled_data = pd.DataFrame(features_scaled, columns=all_features)
    scaled_data[target_column] = target_scaled
    scaled_data.to_csv(scaled_data_file, index=False)
    logging.info(f"Scaled data saved to {scaled_data_file}")

    # Load sequences from .npz
    sequences_data = np.load(sequence_file, allow_pickle=True)
    X = sequences_data['features']
    # Possibly a separate 'labels' but here we do lookahead manually
    lookahead_period = config.get("lookahead_period", 7)
    y = target_scaled[lookahead_period:]
    X = X[:-lookahead_period]

    logging.info(f"Sequences loaded from {sequence_file}. X shape: {X.shape}, y shape: {y.shape}")

    total_samples = len(X)
    train_split = int(config["train_ratio"] * total_samples)
    val_split = int(config["val_ratio"] * total_samples)

    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split : train_split + val_split]
    y_val = y[train_split : train_split + val_split]
    X_test = X[train_split + val_split :]
    y_test = y[train_split + val_split :]

    logging.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Extra scaling on the 3D arrays
    scaler_X = StandardScaler()
    _, _, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    logging.info("Applied StandardScaler to sequences and targets.")
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y


mixed_precision.set_global_policy('mixed_float16')


@log_time
def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape, scaler_y,
                                  tuner_dir='logs', config=None):
    """
    Perform hyperparameter tuning using Keras Tuner (Bayesian Optimization).
    Uses 'val_loss' as the objective by default in this refactoring.
    """
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=config.get("max_trials", 50),
        num_initial_points=config.get("num_initial_points", 10),
        directory=tuner_dir,
        project_name='stock_price_prediction',
        overwrite=True
    )

    stop_early = EarlyStopping(
        monitor='val_loss',
        patience=25,
        mode='min',
        restore_best_weights=True
    )

    combined_callback = CombinedMetricCallback(
        scaler_y,
        config["weight_direction"],
        config["weight_sharpe"]
    )
    combined_callback.validation_data = (X_val, y_val)

    tuner.search(
        X_train, y_train,
        epochs=config.get('tuning_epochs', 10),  # often too short for real training, but good for quick HP search
        validation_data=(X_val, y_val),
        callbacks=[stop_early, combined_callback],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info("Best Hyperparameters (from tuner):")
    for param in best_hps.values:
        logging.info(f" - {param}: {best_hps.get(param)}")

    return best_hps, tuner, combined_callback


@log_time
def train_final_model(X_train, y_train, X_val, y_val, best_hps, input_shape,
                      combined_metric_callback, scaler_y, config):
    """
    Train the final model with best hyperparams from tuner. 
    Resumes from checkpoint if it exists.
    """
    model = build_model(best_hps, input_shape)
    model.summary(print_fn=logging.info)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    epoch_tracker_filepath = os.path.join(config["checkpoint_dir"], 'current_epoch.txt')
    checkpoint_filepath = os.path.join(config["checkpoint_dir"], 'best_model.keras')

    # Re-init combined metric callback
    combined_metric_callback = CombinedMetricCallback(
        scaler_y,
        config["weight_direction"],
        config["weight_sharpe"]
    )
    combined_metric_callback.validation_data = (X_val, y_val)

    initial_lr = best_hps.get('learning_rate')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config["early_stopping_patience"],
        restore_best_weights=True,
        mode='min',
        verbose=1
    )

    callbacks = [
        early_stopping,
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config["reduce_lr_factor"],
            patience=config["reduce_lr_patience"],
            min_lr=config["reduce_lr_min_lr"],
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(
                config["tensorboard_log_dir"],
                datetime.now().strftime("%Y%m%d-%H%M%S")
            ),
            histogram_freq=1,
            update_freq='batch',
            profile_batch=150
        ),
        EpochTracker(epoch_tracker_filepath),
        CosineAnnealingScheduler(
            initial_lr=initial_lr,
            first_restart_epoch=config.get("cosine_annealing_first_restart", 10),
            T_mult=config.get("cosine_annealing_T_mult", 2)
        ),
        combined_metric_callback
    ]

    initial_epoch = 0
    if os.path.exists(checkpoint_filepath):
        logging.info(f"Loading model from checkpoint: {checkpoint_filepath}")
        model = tf.keras.models.load_model(
            checkpoint_filepath,
            custom_objects={'r2_keras': r2_keras},
            compile=False
        )
        optimizer = AdamW(
            learning_rate=best_hps.get('learning_rate'),
            weight_decay=best_hps.get('weight_decay'),
            clipnorm=1.0
        )
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mape', r2_keras])

        if os.path.exists(epoch_tracker_filepath):
            with open(epoch_tracker_filepath, 'r') as f:
                try:
                    initial_epoch = int(f.read())
                    logging.info(f"Resuming training from epoch {initial_epoch}")
                except ValueError:
                    logging.warning("Epoch tracker file corrupted. Starting epoch=0.")
                    initial_epoch = 0
        else:
            logging.warning("No epoch tracker file found. Starting epoch=0.")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1,
        initial_epoch=initial_epoch
    )

    return model, history


@log_time
def evaluate_and_visualize(model, X_test, y_test, scaler_y, config):
    """
    Generate predictions on X_test, calculate metrics, and plot results.
    """
    predictions = model.predict(X_test)
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, 1)

    predictions_rescaled = scaler_y.inverse_transform(predictions)
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    test_mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    test_rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    test_mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)
    test_r2 = r2_score(y_test_rescaled, predictions_rescaled)

    # Directional Accuracy
    direction_true = np.sign(y_test_rescaled[1:] - y_test_rescaled[:-1])
    direction_pred = np.sign(predictions_rescaled[1:] - predictions_rescaled[:-1])
    directional_accuracy = np.mean(direction_true == direction_pred) * 100

    # Sharpe Ratio
    daily_returns = (predictions_rescaled[1:] - predictions_rescaled[:-1]) / predictions_rescaled[:-1]
    if np.std(daily_returns) != 0:
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    cumulative_returns = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    # Combined Metric
    test_combined_metric = combined_metric(
        y_test, predictions, scaler_y,
        config["weight_direction"], config["weight_sharpe"]
    )

    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    logging.info(f"Test MAPE: {test_mape:.4f}")
    logging.info(f"Test R²: {test_r2:.4f}")
    logging.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    logging.info(f"Max Drawdown: {max_drawdown:.4f}")
    logging.info(f"Test Combined Metric: {test_combined_metric:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_rescaled, label='Actual', color='blue')
    plt.plot(predictions_rescaled, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Residual Plot
    residuals = y_test_rescaled - predictions_rescaled
    plt.figure(figsize=(14, 6))
    plt.scatter(predictions_rescaled, residuals, alpha=0.5, c='green')
    plt.hlines(y=0, xmin=min(predictions_rescaled), xmax=max(predictions_rescaled), colors='r')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.show()

    return {
        'MAE': test_mae,
        'RMSE': test_rmse,
        'MAPE': test_mape,
        'R2': test_r2,
        'Directional Accuracy': directional_accuracy,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Combined Metric': test_combined_metric
    }


@log_time
def plot_training_history(history_file: str) -> None:
    """
    Plot training/validation metrics from a JSON history file.
    """
    if not os.path.exists(history_file):
        logging.error(f"History file not found: {history_file}")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['mae']) + 1)

    plt.figure(figsize=(18, 5))

    # 1) MAE
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['mae'], 'b-', label='Train MAE')
    plt.plot(epochs, history['val_mae'], 'r-', label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # 2) MAPE
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['mape'], 'b-', label='Train MAPE')
    plt.plot(epochs, history['val_mape'], 'r-', label='Val MAPE')
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.legend()

    # 3) R²
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['r2_keras'], 'b-', label='Train R²')
    plt.plot(epochs, history['val_r2_keras'], 'r-', label='Val R²')
    plt.title('R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()

    # 4) Loss
    #   If you want to see combined_metric, rename keys or store them differently.
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


@log_time
def train_model(config: dict) -> None:
    """
    Main training function:
      1) Load & process data
      2) Hyperparameter tune
      3) Train final model
      4) Save model + scalers
    """
    # Load Config
    config = load_config('config.json')

    # Validate files
    validate_files([config["input_file"], config["sequence_file"]])

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_data(
        config["input_file"],
        config["sequence_file"],
        config["feature_columns"],
        config["target_column"],
        config
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    # Tuning
    best_hps, tuner, combined_cb = perform_hyperparameter_tuning(
        X_train, y_train, X_val, y_val, input_shape, scaler_y,
        tuner_dir='logs', config=config
    )

    # Final train
    final_model, history = train_final_model(
        X_train, y_train, X_val, y_val,
        best_hps, input_shape, combined_cb, scaler_y, config
    )

    # Save model
    os.makedirs(os.path.dirname(config["output_model_file"]), exist_ok=True)
    final_model.save(config["output_model_file"])
    logging.info(f"Model saved to {config['output_model_file']}")

    # Save scalers
    scaler_dir = config.get("scaler_dir", "data/scalers")
    scaler_X_filename = os.path.join(scaler_dir, "scaler_X.save")
    scaler_y_filename = os.path.join(scaler_dir, "scaler_y.save")
    save_scalers(scaler_X, scaler_y, scaler_X_filename, scaler_y_filename)

    # Optionally save training history
    if "history_file" in config:
        history_dict = history.history
        os.makedirs(os.path.dirname(config["history_file"]), exist_ok=True)
        with open(config["history_file"], 'w') as f:
            json.dump(history_dict, f, indent=2)
        logging.info(f"Training history saved to {config['history_file']}")

    logging.info("Training completed successfully.")


if __name__ == "__main__":
    train_model(args.config)
