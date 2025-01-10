#!/usr/bin/env python
#======================================
#  train.py
#======================================

import os
import json
import logging
from datetime import datetime

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

from utils import (
    EpochTracker, CosineAnnealingScheduler, CombinedMetricCallback, r2_keras,
    load_config, load_data, build_model, combined_metric, scale_data, save_scalers, log_time, validate_files
)

# Configuration and Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)

config = load_config('config.json')
required_keys = [
    "input_file", "sequence_file", "feature_columns", "target_column", "train_ratio",
    "val_ratio", "epochs", "batch_size", "checkpoint_dir",
    "output_model_file", "history_file", "tensorboard_log_dir",
    "early_stopping_patience", "reduce_lr_factor",
    "reduce_lr_patience", "reduce_lr_min_lr",
    "cosine_annealing_first_restart", "cosine_annealing_T_mult",
    "feature_engineering_window",
    "max_trials",
    "num_initial_points",
    "lookahead_period",
    "weight_direction",
    "weight_sharpe",
    "scaler_dir"
]

missing_keys = [key for key in required_keys if key not in config]
if missing_keys:
    raise ValueError(f"Missing configuration parameters: {missing_keys}")

mixed_precision.set_global_policy('mixed_float16')

# Hyperparameter Tuning with Bayesian Optimization
@log_time
def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape, scaler_y, tuner_dir='logs'):
    """
    Perform hyperparameter tuning using Keras Tuner with Bayesian Optimization.
    """
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective=kt.Objective("val_combined_metric", direction="max"),
        max_trials=config.get("max_trials", 50),
        num_initial_points=config.get("num_initial_points", 10),
        directory=tuner_dir,
        project_name='stock_price_prediction',
        overwrite=True
    )

    stop_early = EarlyStopping(
        monitor='val_combined_metric',
        patience=25,
        mode='max',
        restore_best_weights=True
    )

    combined_metric_callback = CombinedMetricCallback(scaler_y, config["weight_direction"], config["weight_sharpe"])
    combined_metric_callback.validation_data = (X_val, y_val)

    tuner.search(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[stop_early, combined_metric_callback],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    logging.info("Best Hyperparameters:")
    for param in best_hps.values:
        logging.info(f" - {param}: {best_hps.get(param)}")

    return best_hps, tuner, combined_metric_callback

# Retraining the Model with Best Hyperparameters
@log_time
def train_final_model(X_train, y_train, X_val, y_val, best_hps, input_shape, combined_metric_callback, scaler_y, config):
    """
    Train the final model using the best hyperparameters.
    """
    model = build_model(best_hps, input_shape)
    model.summary(print_fn=logging.info)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    epoch_tracker_filepath = os.path.join(config["checkpoint_dir"], 'current_epoch.txt')
    checkpoint_filepath = os.path.join(config["checkpoint_dir"], 'best_model.keras')
    combined_metric_callback = CombinedMetricCallback(scaler_y, config["weight_direction"], config["weight_sharpe"])
    combined_metric_callback.validation_data = (X_val, y_val)

    initial_lr = best_hps.get('learning_rate')

    early_stopping = EarlyStopping(
        monitor='val_combined_metric',
        patience=config["early_stopping_patience"],
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    callbacks = [
        early_stopping,
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_combined_metric',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_combined_metric',
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
        model.compile(
            optimizer=optimizer,
            loss='mae',
            metrics=['mae', 'mape', r2_keras]
        )

        if os.path.exists(epoch_tracker_filepath):
            with open(epoch_tracker_filepath, 'r') as f:
                try:
                    initial_epoch = int(f.read())
                    logging.info(f"Resuming training from epoch {initial_epoch}")
                except ValueError:
                    logging.warning(
                        "Epoch tracker file is corrupted. Starting from epoch 0.")
                    initial_epoch = 0
        else:
            logging.warning(
                "Epoch tracker file not found. Starting from epoch 0.")
            initial_epoch = 0
    else:
        logging.info("No checkpoint found. Training from scratch.")
        initial_epoch = 0

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1,
        initial_epoch=0
    )

    return model, history

@log_time
def evaluate_and_visualize(model, X_test, y_test, scaler_y, config):
    """
    Generate predictions, calculate metrics (including combined_metric), and visualize results.
    """
    predictions = model.predict(X_test)

    # Reshape predictions if needed
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, 1)

    predictions_rescaled = scaler_y.inverse_transform(predictions)
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    test_mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    test_rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    test_mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)
    test_r2 = r2_score(y_test_rescaled, predictions_rescaled)

    # Calculate Directional Accuracy
    direction_true = np.sign(y_test_rescaled[1:] - y_test_rescaled[:-1])
    direction_pred = np.sign(predictions_rescaled[1:] - predictions_rescaled[:-1])
    directional_accuracy = np.mean(direction_true == direction_pred) * 100

    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    daily_returns = (predictions_rescaled[1:] - predictions_rescaled[:-1]) / predictions_rescaled[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized

    # Calculate Maximum Drawdown
    cumulative_returns = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    # Calculate the combined metric (using the function from utils.py)
    test_combined_metric = combined_metric(
        y_test, predictions, scaler_y,
        config["weight_direction"], config["weight_sharpe"]
    )

    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    logging.info(f"Test MAPE: {test_mape:.4f}%")
    logging.info(f"Test R²: {test_r2:.4f}")
    logging.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    logging.info(f"Max Drawdown: {max_drawdown:.4f}")
    logging.info(f"Test Combined Metric: {test_combined_metric:.4f}")

    # Visualization: Predictions vs Actual
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_rescaled, label='Actual', color='blue')
    plt.plot(predictions_rescaled, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Residual Analysis
    residuals = y_test_rescaled - predictions_rescaled
    plt.figure(figsize=(14, 6))
    plt.scatter(predictions_rescaled, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(predictions_rescaled), xmax=max(predictions_rescaled), colors='r')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Values')
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
def plot_training_history(history_file):
    """
    Plot training and validation metrics from the history file.

    Parameters
    ----------
    history_file : str
        Path to the JSON file containing training history.
    """
    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['mae']) + 1)

    plt.figure(figsize=(18, 5))

    # Plot MAE
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['mae'], 'b-', label='Training MAE')
    plt.plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plot MAPE
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['mape'], 'b-', label='Training MAPE')
    plt.plot(epochs, history['val_mape'], 'r-', label='Validation MAPE')
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE (%)')
    plt.legend()

    # Plot R² Score
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['r2_keras'], 'b-', label='Training R²')
    plt.plot(epochs, history['val_r2_keras'], 'r-', label='Validation R²')
    plt.title('R² Score')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()

    # Plot Combined Metric
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['combined_metric'], 'b-', label='Training Combined Metric')
    plt.plot(epochs, history['val_combined_metric'], 'r-', label='Validation Combined Metric')
    plt.title('Combined Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Combined')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Training Function
@log_time
def train_model(config):
    """
    Train the model with hyperparameter tuning and save results.
    """
    try:
        # Validate the existence of necessary files before starting
        validate_files([config["input_file"], config["sequence_file"]])

        # Load and preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_data(
            config["input_file"],
            config["sequence_file"],
            config["feature_columns"],
            config["target_column"],
            config
        )

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Perform hyperparameter tuning
        best_hps, tuner, combined_metric_callback = perform_hyperparameter_tuning(
            X_train, y_train, X_val, y_val, input_shape, scaler_y, tuner_dir='logs'
        )

        # Train the final model
        final_model, history = train_final_model(
            X_train, y_train, X_val, y_val, best_hps, input_shape, combined_metric_callback, scaler_y, config
        )

        # Save the trained model
        os.makedirs(os.path.dirname(config["output_model_file"]), exist_ok=True)
        final_model.save(config["output_model_file"])
        logging.info(f"Model saved to {config['output_model_file']}")

        # Save scalers
        scaler_dir = config.get("scaler_dir","scalers")
        scaler_X_filename = os.path.join(config["scaler_dir"], "scaler_X.save")
        scaler_y_filename = os.path.join(config["scaler_dir"], "scaler_y.save")
        save_scalers(scaler_X, scaler_y, scaler_X_filename, scaler_y_filename)

        logging.info("Training completed successfully.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

# Entry Point
if __name__ == "__main__":
    train_model(config)
