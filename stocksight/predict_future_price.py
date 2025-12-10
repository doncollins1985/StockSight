#================================
# model/predict_future_price.py
#================================

import joblib
import pandas as pd
import numpy as np
import torch
from .utils import validate_files, log_time, load_config, load_scalers, get_device
from .models import StockPredictor
from datetime import datetime
import logging
from pandas.tseries.offsets import BDay

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@log_time
def predict_next_days_with_sentiment(
    input_file: str,
    model_file: str,
    scaler_file_X: str,
    scaler_file_y: str,
    feature_columns: list,
    window_size: int,
    num_days: int,
    target_column: str = 'Close'
) -> pd.DataFrame:
    """
    Predict future stock prices including sentiment features.
    """
    try:
        validate_files([input_file, model_file, scaler_file_X, scaler_file_y])
        data = pd.read_csv(input_file, parse_dates=['Date'])
        required_columns = ['Date', 'Close'] + feature_columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")
        if len(data) < window_size:
            raise ValueError("Not enough historical data for predictions.")
        
        feature_scaler, close_scaler = load_scalers(scaler_file_X, scaler_file_y)
        
        # Load Model
        device = get_device()
        logging.info(f"Loading model from {model_file} to {device}")
        checkpoint = torch.load(model_file, map_location=device)
        
        # Reconstruct model
        model = StockPredictor(checkpoint['input_shape'], checkpoint['hyperparameters'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        logging.info("Model loaded successfully.")
        predictions = []
        current_data = data[feature_columns].values[-window_size:]
        last_date = data['Date'].iloc[-1]
        last_price = data['Close'].iloc[-1]
        
        # Infer time step from data
        if len(data) >= 2:
            step_delta = data['Date'].diff().mode()[0]
        else:
            step_delta = BDay(1)
            
        logging.info(f"Inferred prediction time step: {step_delta}")

        default_sentiment = {
            'Positive': 0,
            'Negative': 0,
            'Neutral': 1,
            'Aggregate_Score': 0
        }
        
        for day in range(1, num_days + 1):
            current_data_scaled = feature_scaler.transform(current_data)
            # Shape: [1, seq_len, features]
            current_data_tensor = torch.tensor(current_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predicted_scaled = model(current_data_tensor).cpu().numpy()
            
            prediction_raw = close_scaler.inverse_transform(predicted_scaled)[0, 0]
            
            if target_column == 'Log_Return':
                # Reconstruct price from log return
                # P_next = P_prev * exp(log_return)
                predicted_price = last_price * np.exp(prediction_raw)
                predicted_log_return = prediction_raw
            else:
                predicted_price = prediction_raw
                predicted_log_return = np.log(predicted_price / last_price) if last_price != 0 else 0
            
            next_date = last_date + step_delta
            pct_change = ((predicted_price - last_price) / last_price * 100) if last_price != 0 else 0.0
            predictions.append({
                'Date': next_date, # Keep as timestamp
                'Predicted_Price': predicted_price,
                'Pct_Change': pct_change
            })
            logging.debug(f"Step {day}: {next_date}, Predicted Price: {predicted_price:.2f}, Pct Change: {pct_change:.2f}%")
            last_date = next_date
            last_price = predicted_price
            
            # Update window for next prediction
            next_features = current_data[-1].copy()
            
            # Update 'Close' column
            if 'Close' in feature_columns:
                close_idx = feature_columns.index('Close')
                next_features[close_idx] = predicted_price

            # Update 'Log_Return' column if it exists
            if 'Log_Return' in feature_columns:
                lr_idx = feature_columns.index('Log_Return')
                next_features[lr_idx] = predicted_log_return
            
            # Update 'Close_Lag_1' column if it exists
            if 'Close_Lag_1' in feature_columns:
                lag_idx = feature_columns.index('Close_Lag_1')
                # Close_Lag_1 for next step is the Close of the current step (predicted_price)
                # Wait, 'current_data[-1]' is the feature vector at T.
                # 'next_features' will be feature vector at T+1.
                # Close at T+1 is unknown (we just predicted it, so we use it).
                # Close_Lag_1 at T+1 is Close at T.
                # Close at T was 'predicted_price' (from loop context) or 'last_price' before update.
                # 'predicted_price' is Close at T+1? No.
                # Let's align time.
                # Input: Features at [T-W, ..., T-1]. Model predicts Target at T.
                # Target at T is Log_Return between T-1 and T.
                # So we get Price at T.
                # Next input sequence should be [T-W+1, ..., T].
                # Features at T:
                # Close: Price at T.
                # Log_Return: log(P_T / P_{T-1}).
                # Close_Lag_1: P_{T-1}.
                
                # So next_features (at T) should have:
                # Close = predicted_price (P_T)
                # Log_Return = predicted_log_return (R_T)
                # Close_Lag_1 = P_{T-1} (which was 'last_price' BEFORE update, i.e., Price at T-1)
                
                # 'last_price' variable tracks the most recent known price.
                # At start of loop: last_price = P_{T-1}.
                # We predict R_T.
                # predicted_price = P_T = P_{T-1} * exp(R_T).
                # next_features (for T):
                #   Close = P_T
                #   Log_Return = R_T
                #   Close_Lag_1 = P_{T-1}
                # But 'next_features' was copied from `current_data[-1]` which was features at T-1.
                # So we update it to represent T.
                # P_{T-1} is what `current_data[-1]` had in 'Close'.
                
                prev_close_val = current_data[-1][feature_columns.index('Close')]
                next_features[lag_idx] = prev_close_val

            # Set sentiment columns to neutral defaults for future dates
            for sentiment_col, default_val in default_sentiment.items():
                if sentiment_col in feature_columns:
                    idx = feature_columns.index(sentiment_col)
                    next_features[idx] = default_val
            
            # Shift window: remove first (oldest), append new
            current_data = np.vstack([current_data[1:], next_features])

        predictions_df = pd.DataFrame(predictions)
        logging.info("Prediction complete.")
        return predictions_df
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise


def main(config_path: str = "config.json", num_days: int = 5) -> None:
    """
    Main function to predict future stock prices using the provided configuration.
    """
    try:
        config = load_config(config_path)
        predictions = predict_next_days_with_sentiment(
            input_file=config['merged_file'],
            model_file=config['output_model_file'],
            scaler_file_X=config['scaler_file_X'],
            scaler_file_y=config['scaler_file_y'],
            feature_columns=config['feature_columns'],
            window_size=config['window_size'],
            num_days=num_days,
            target_column=config.get('target_column', 'Close')
        )
        predictions.to_csv("predictions_with_sentiment.csv", index=False)
        logging.info("Predictions saved to predictions_with_sentiment.csv.")
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()