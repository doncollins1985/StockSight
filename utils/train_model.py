# model/train_model.py

import os
import json
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    LayerNormalization, Bidirectional, Attention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import keras_tuner as kt  # Ensure Keras Tuner is installed: pip install keras-tuner
from utils.utils import load_config

# Enable mixed precision for faster training on compatible hardware
mixed_precision.set_global_policy('mixed_float16')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


class EpochTracker(tf.keras.callbacks.Callback):
    """
    Custom callback to track and save the current epoch number.
    """

    def __init__(self, filepath: str):
        super(EpochTracker, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch: int, logs=None):
        with open(self.filepath, 'w') as f:
            f.write(str(epoch + 1))


def load_data(input_file: str, sequence_file: str, feature_columns: list, config: dict):
    """
    Load and prepare the data for training, validation, and testing.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not os.path.exists(sequence_file):
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

    # Load raw data
    data = pd.read_csv(input_file)
    logging.info(f"Data loaded from {input_file}, shape: {data.shape}")

    # Load preprocessed sequences
    sequences_data = np.load(sequence_file, allow_pickle=True)
    X = sequences_data['features']
    y = sequences_data['labels']
    logging.info(f"Sequences loaded from {sequence_file}. X shape: {X.shape}, y shape: {y.shape}")

    total_samples = len(X)
    train_split = int(config["train_ratio"] * total_samples)
    val_split = int(config["val_ratio"] * total_samples)

    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:train_split + val_split]
    y_val = y[train_split:train_split + val_split]
    X_test = X[train_split + val_split:]
    y_test = y[train_split + val_split:]

    # Scale features
    scaler_X = MinMaxScaler()
    num_samples, window, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    logging.info("Feature scaling applied using MinMaxScaler.")

    # Scale targets
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    logging.info("Target scaling applied using MinMaxScaler.")

    logging.info(f"Training samples: {len(X_train_scaled)}")
    logging.info(f"Validation samples: {len(X_val_scaled)}")
    logging.info(f"Testing samples: {len(X_test_scaled)}")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y


def build_model(hp, input_shape: tuple) -> tf.keras.Model:
    """
    Build and compile a Keras model with hyperparameters.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Convolutional layers
    num_conv_layers = hp.Int('num_conv_layers', 1, 2, default=2)
    for i in range(num_conv_layers):
        filters = hp.Int(f'conv_filters_{i}', min_value=32, max_value=256, step=32, default=64)
        kernel_size = hp.Int(f'conv_kernel_size_{i}', min_value=2, max_value=5, step=1, default=3)
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(
                hp.Float(f'conv_l2_reg_{i}', 1e-5, 1e-3, sampling='log', default=1e-4)
            )
        )(x)
        x = LayerNormalization()(x)
        dropout_rate = hp.Float(f'conv_dropout_rate_{i}', 0.1, 0.5, step=0.1, default=0.3)
        x = Dropout(rate=dropout_rate)(x)
        x = MaxPooling1D(pool_size=2)(x)

    # LSTM layers
    num_layers = hp.Int('num_layers', 1, 3, default=2)
    lstm_layers = []
    for i in range(num_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=64, max_value=512, step=64, default=128)
        return_sequences = True if num_layers > 1 else False
        x_lstm = Bidirectional(
            LSTM(
                units=units,
                activation='tanh',
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.l2(
                    hp.Float(f'l2_reg_{i}', 1e-5, 1e-3, sampling='log', default=1e-4)
                )
            )
        )(x)
        lstm_layers.append(x_lstm)
        x = LayerNormalization()(x_lstm)
        dropout_rate = hp.Float(f'lstm_dropout_rate_{i}', 0.2, 0.5, step=0.1, default=0.3)
        x = Dropout(rate=dropout_rate)(x)

    # Attention mechanism if multiple LSTM layers
    if num_layers > 1:
        attn = Attention()([x_lstm, x_lstm])
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x)

    # Dense layers
    num_dense_layers = hp.Int('num_dense_layers', 1, 3, default=2)
    for j in range(num_dense_layers):
        units = hp.Int(f'dense_units_{j}', min_value=16, max_value=128, step=16, default=64)
        x = Dense(units=units, activation='relu')(x)
        x = LayerNormalization()(x)
        dropout_rate = hp.Float(f'dense_dropout_rate_{j}', 0.2, 0.5, step=0.1, default=0.3)
        x = Dropout(rate=dropout_rate)(x)

    # Output layer
    outputs = Dense(1, dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    return model


def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape: tuple, tuner_dir: str = 'kt_logs'):
    """
    Perform hyperparameter tuning using Keras Tuner with Bayesian Optimization.
    """
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=30,
        num_initial_points=10,
        directory=tuner_dir,
        project_name='finbert_lstm_tuning_bayesian',
        overwrite=False,
        seed=SEED
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[stop_early],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info("Best Hyperparameters:")
    for param in best_hps.values:
        logging.info(f" - {param}: {best_hps.get(param)}")
    return best_hps, tuner


def train_final_model(X_train, y_train, X_val, y_val, best_hps, input_shape: tuple, config: dict):
    """
    Train the final model using the best hyperparameters.
    """
    model = build_model(best_hps, input_shape)
    model.summary(print_fn=lambda x: logging.info(x))

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    epoch_tracker_filepath = os.path.join(config["checkpoint_dir"], 'current_epoch.txt')
    checkpoint_filepath = os.path.join(config["checkpoint_dir"], 'finbert_lstm_model_best.keras')

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
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
            profile_batch=50
        ),
        EpochTracker(epoch_tracker_filepath)
    ]

    initial_lr = best_hps.get('learning_rate')

    def warmup_scheduler(epoch, lr):
        if epoch < config["warmup_epochs"]:
            return initial_lr * (epoch + 1) / config["warmup_epochs"]
        return lr

    callbacks.append(LearningRateScheduler(warmup_scheduler, verbose=1))

    if os.path.exists(checkpoint_filepath):
        logging.info(f"Loading model from checkpoint: {checkpoint_filepath}")
        model = tf.keras.models.load_model(checkpoint_filepath)
        if os.path.exists(epoch_tracker_filepath):
            try:
                with open(epoch_tracker_filepath, 'r') as f:
                    initial_epoch = int(f.read())
                logging.info(f"Resuming training from epoch {initial_epoch}")
            except ValueError:
                logging.warning("Epoch tracker file is corrupted. Starting from epoch 0.")
                initial_epoch = 0
        else:
            logging.warning("Epoch tracker file not found. Starting from epoch 0.")
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
        initial_epoch=initial_epoch
    )
    return model, history


def train_model_with_sentiment(config: dict) -> None:
    """
    Train the model with hyperparameter tuning and technical indicators, then save results.
    """
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_data(
            config["merged_file"],
            config["sequence_file"],
            config["feature_columns"],
            config
        )

        input_shape = (X_train.shape[1], X_train.shape[2])

        best_hps, tuner = perform_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape, tuner_dir='kt_logs')

        final_model, history = train_final_model(X_train, y_train, X_val, y_val, best_hps, input_shape, config)

        test_predictions = final_model.predict(X_test)
        test_predictions_rescaled = scaler_y.inverse_transform(test_predictions)
        y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        test_mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
        test_mape = mean_absolute_percentage_error(y_test_rescaled, test_predictions_rescaled)
        test_r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

        logging.info(f"Test MAE: {test_mae:.4f}")
        logging.info(f"Test MAPE: {test_mape:.4f}%")
        logging.info(f"Test R²: {test_r2:.4f}")

        os.makedirs(os.path.dirname(config["output_model_file"]), exist_ok=True)
        final_model.save(config["output_model_file"])
        logging.info(f"Model saved to {config['output_model_file']}")

        history_dict = history.history
        history_dict.update({'test_mae': test_mae, 'test_mape': test_mape, 'test_r2': test_r2})
        os.makedirs(os.path.dirname(config["history_file"]), exist_ok=True)
        with open(config["history_file"], 'w') as f:
            json.dump(history_dict, f, indent=4)
        logging.info(f"Training history saved to {config['history_file']}")

        scaler_X_filename = os.path.join(os.path.dirname(config["history_file"]), "scaler_X.save")
        scaler_y_filename = os.path.join(os.path.dirname(config["history_file"]), "scaler_y.save")
        joblib.dump(scaler_X, scaler_X_filename)
        joblib.dump(scaler_y, scaler_y_filename)
        logging.info(f"Scalers saved to {scaler_X_filename} and {scaler_y_filename}")

        logging.info("Training completed successfully.")

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        raise


def main(config_path: str = "config.json") -> None:
    """
    Main function to train the model using the provided configuration.
    """
    try:
        config = load_config(config_path)
        train_model_with_sentiment(config)
    except Exception as e:
        logging.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()

