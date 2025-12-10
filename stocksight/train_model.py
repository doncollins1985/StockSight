# model/train_model.py

import os
import json
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

from .utils import load_config, save_scalers, get_device
from .models import StockPredictor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
    scaler_X = StandardScaler()
    num_samples, window, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    logging.info("Feature scaling applied using StandardScaler.")

    # Scale targets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    logging.info("Target scaling applied using StandardScaler.")

    logging.info(f"Training samples: {len(X_train_scaled)}")
    logging.info(f"Validation samples: {len(X_val_scaled)}")
    logging.info(f"Testing samples: {len(X_test_scaled)}")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y


def calculate_ctm_loss(outputs, sigmas, targets, inputs, direction_weight=1.0):
    """
    Calculates CTM loss (Min Loss + Min Sigma Loss) / 2 + Directional Penalty.
    Uses NLL (Gaussian) as the base loss.
    
    outputs: (B, T, 1)
    sigmas: (B, T, 1)
    targets: (B, 1) or (B)
    inputs: (B, Seq, F)
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(1) # (B, 1)
        
    batch_size, n_ticks, _ = outputs.shape
    targets_exp = targets.unsqueeze(1).expand(-1, n_ticks, -1) # (B, T, 1)
    
    # 1. Compute NLL Loss per tick
    # NLL = 0.5 * log(sigma^2) + 0.5 * (y - pred)^2 / sigma^2
    # clamp sigma for stability
    sigmas = torch.clamp(sigmas, min=1e-6)
    var = sigmas ** 2
    nll = 0.5 * torch.log(var) + 0.5 * (targets_exp - outputs)**2 / var
    loss_per_tick = nll.squeeze(-1) # (B, T)
    
    # 2. Select ticks
    # Best Loss (Minimum NLL)
    min_loss_vals, _ = torch.min(loss_per_tick, dim=1) # (B,)
    
    # Best Certainty (Minimum Sigma)
    # Use average sigma over time or instantaneous? Paper uses C^t. 
    # Here we use instantaneous sigma.
    sigmas_s = sigmas.squeeze(-1) # (B, T)
    min_sigma_vals, min_sigma_idx = torch.min(sigmas_s, dim=1) # (B,)
    
    # Loss at best certainty
    loss_at_best_cert = torch.gather(loss_per_tick, 1, min_sigma_idx.unsqueeze(1)).squeeze(1)
    
    # Average
    main_loss = (min_loss_vals + loss_at_best_cert) / 2.0
    main_loss = main_loss.mean()
    
    # 3. Directional Penalty
    # We apply it to the prediction at the 'best certainty' tick.
    current_price = inputs[:, -1, 0].unsqueeze(1) # (B, 1)
    best_out = torch.gather(outputs, 1, min_sigma_idx.unsqueeze(1).unsqueeze(2)).squeeze(1) # (B, 1)
    
    target_diff = targets - current_price
    pred_diff = best_out - current_price
    
    # Penalize if signs differ
    direction_loss = torch.mean(torch.relu(-1.0 * target_diff * pred_diff))
    
    return main_loss + (direction_weight * direction_loss)


def train_one_epoch(model, dataloader, criterion, optimizer, device, direction_weight=1.0):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # CTM Forward pass (all ticks)
        outputs, sigmas = model(inputs, return_all_ticks=True)
        
        loss = calculate_ctm_loss(outputs, sigmas, targets, inputs, direction_weight)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device, direction_weight=1.0):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, sigmas = model(inputs, return_all_ticks=True)
            loss = calculate_ctm_loss(outputs, sigmas, targets, inputs, direction_weight)
            
            running_loss += loss.item() * inputs.size(0)
            
    return running_loss / len(dataloader.dataset)


def objective(trial, X_train, y_train, X_val, y_val, input_shape, device):
    # Hyperparameters for CTM
    hp = {
        'd_model': trial.suggest_int('d_model', 64, 256, step=32),
        'memory_length': trial.suggest_int('memory_length', 5, 20),
        'n_ticks': trial.suggest_int('n_ticks', 3, 10),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'nlm_hidden': trial.suggest_int('nlm_hidden', 16, 64),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'j_out': trial.suggest_int('j_out', 16, 64, step=16),
        'j_action': trial.suggest_int('j_action', 16, 64, step=16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    model = StockPredictor(input_shape, hp).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    criterion = None # Not used for CTM training

    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(10): 
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss


def train_final_model(X_train, y_train, X_val, y_val, best_hps, input_shape, config, device):
    model = StockPredictor(input_shape, best_hps).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=best_hps['learning_rate'], weight_decay=best_hps['weight_decay'])
    criterion = None 
    
    batch_size = config.get("batch_size", 32)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config["reduce_lr_factor"], 
        patience=config["reduce_lr_patience"], min_lr=config["reduce_lr_min_lr"]
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'ctm_model_best.pth')

    epochs = config["epochs"]
    
    logging.info(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                logging.info("Early stopping triggered.")
                break
                
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history


def get_default_hyperparameters(config):
    """
    Returns default hyperparameters for CTM, overridden by config if keys exist.
    """
    defaults = {
        'd_model': 128,
        'memory_length': 10,
        'n_ticks': 5,
        'n_heads': 4,
        'nlm_hidden': 32,
        'dropout': 0.1,
        'j_out': 32,
        'j_action': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    }
    
    for key in defaults:
        if key in config:
            defaults[key] = config[key]
            
    return defaults


def train_model_with_sentiment(config: dict, use_tuning: bool = False) -> None:
    """
    Train the model with optional hyperparameter tuning and technical indicators, then save results.
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_data(
            config["merged_file"],
            config["sequence_file"],
            config["feature_columns"],
            config
        )

        input_shape = (X_train.shape[1], X_train.shape[2])

        if use_tuning:
            # Hyperparameter Tuning
            logging.info("Starting hyperparameter tuning...")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_shape, device), n_trials=30)
            
            best_hps = study.best_params
            logging.info("Best Hyperparameters from Tuning:")
            for key, value in best_hps.items():
                logging.info(f" - {key}: {value}")
        else:
            logging.info("Using default hyperparameters (tuning disabled).")
            best_hps = get_default_hyperparameters(config)
            logging.info("Hyperparameters:")
            for key, value in best_hps.items():
                logging.info(f" - {key}: {value}")

        # Train Final Model
        final_model, history = train_final_model(X_train, y_train, X_val, y_val, best_hps, input_shape, config, device)

        # Evaluation
        final_model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            # For inference evaluation, use default forward which selects the final tick
            test_predictions = final_model(X_test_tensor, return_all_ticks=False).cpu().numpy()
            
        test_predictions_rescaled = scaler_y.inverse_transform(test_predictions)
        y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        test_mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
        test_mape = mean_absolute_percentage_error(y_test_rescaled, test_predictions_rescaled)
        test_r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

        logging.info(f"Test MAE: {test_mae:.4f}")
        logging.info(f"Test MAPE: {test_mape:.4f}%")
        logging.info(f"Test R²: {test_r2:.4f}")

        # Save Final Model with Hyperparams
        os.makedirs(os.path.dirname(config["output_model_file"]), exist_ok=True)
        save_dict = {
            'state_dict': final_model.state_dict(),
            'hyperparameters': best_hps,
            'input_shape': input_shape
        }
        torch.save(save_dict, config["output_model_file"])
        logging.info(f"Model saved to {config['output_model_file']}")

        # Save History
        history_dict = history
        history_dict.update({'test_mae': test_mae, 'test_mape': test_mape, 'test_r2': test_r2})
        os.makedirs(os.path.dirname(config["history_file"]), exist_ok=True)
        with open(config["history_file"], 'w') as f:
            json.dump(history_dict, f, indent=4)
        logging.info(f"Training history saved to {config['history_file']}")

        # Save scalers
        save_scalers(scaler_X, scaler_y, config["scaler_file_X"], config["scaler_file_y"])

        logging.info("Training completed successfully.")

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        raise


def main(config_path: str = "config.json", use_tuning: bool = False) -> None:
    try:
        config = load_config(config_path)
        train_model_with_sentiment(config, use_tuning=use_tuning)
    except Exception as e:
        logging.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()