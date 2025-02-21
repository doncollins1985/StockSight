#================================
# model/predict_future_price.py
#================================

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.utils import validate_files, log_time, load_config
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
    num_days: int
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
        feature_scaler = joblib.load(scaler_file_X)
        close_scaler = joblib.load(scaler_file_y)
        logging.info("Scalers loaded successfully.")
        model = tf.keras.models.load_model(model_file)
        logging.info("Model loaded successfully.")
        predictions = []
        current_data = data[feature_columns].values[-window_size:]
        last_date = data['Date'].iloc[-1]
        last_price = data['Close'].iloc[-1]
        default_sentiment = {
            'Positive': 0,
            'Negative': 0,
            'Neutral': 1,
            'Aggregate_Score': 0
        }
        for day in range(1, num_days + 1):
            current_data_scaled = feature_scaler.transform(current_data)
            current_data_scaled = current_data_scaled.reshape(1, window_size, -1)
            predicted_scaled = model.predict(current_data_scaled, verbose=0)
            predicted_price = close_scaler.inverse_transform(predicted_scaled)[0, 0]
            next_date = last_date + BDay(1)
            pct_change = ((predicted_price - last_price) / last_price * 100) if last_price != 0 else 0.0
            predictions.append({
                'Date': next_date.date(),
                'Predicted_Price': predicted_price,
                'Pct_Change': pct_change
            })
            logging.debug(f"Day {day}: {next_date.date()}, Predicted Price: {predicted_price:.2f}, Pct Change: {pct_change:.2f}%")
            last_date = next_date
            last_price = predicted_price
            current_data = np.roll(current_data, -1, axis=0)
            close_idx = feature_columns.index('Close')
            current_data[-1, close_idx] = predicted_price
            for sentiment_col, default_val in default_sentiment.items():
                if sentiment_col in feature_columns:
                    idx = feature_columns.index(sentiment_col)
                    current_data[-1, idx] = default_val
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
            feature_columns=config['feature_columns'] + ['Positive', 'Negative', 'Neutral', 'Aggregate_Score'],
            window_size=config['window_size'],
            num_days=num_days
        )
        predictions.to_csv("predictions_with_sentiment.csv", index=False)
        logging.info("Predictions saved to predictions_with_sentiment.csv.")
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

