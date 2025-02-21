#================================
# model/evaluate_model.py
#================================

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from utils.utils import log_time, load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/evaluation.log"),
            logging.StreamHandler()
        ]
    )


def load_model_and_data(config):
    """
    Load the trained model and test data.
    """
    try:
        model_path = config['output_model_file']
        logging.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    try:
        sequence_file = config["sequence_file"]
        logging.info(f"Loading data from {sequence_file}")
        sequences = np.load(sequence_file, allow_pickle=True)
        X_test, y_test = sequences["features"], sequences["labels"]
        logging.info(f"Loaded {X_test.shape[0]} test samples.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise
    if X_test.shape[0] != y_test.shape[0]:
        error_msg = "Mismatch between number of features and labels."
        logging.error(error_msg)
        raise ValueError(error_msg)
    return model, X_test, y_test

def load_scaler_X(config):
    """
    Load the feature scaler used during training.
    """
    try:
        scaler_dir = os.path.dirname(config["scaler_dir"])
        scaler_X_file = os.path.join(scaler_dir, "scaler_X.save")
        logging.info(f"Loading target scaler from {scaler_X_file}")
        scaler_X = joblib.load(scaler_X_file)
        return scaler_X
    except Exception as e:
        logging.error(f"Failed to load feature scaler: {e}")
        raise

def load_scaler_y(config):
    """
    Load the target scaler used during training.
    """
    try:
        scaler_dir = os.path.dirname(config["scaler_dir"])
        scaler_y_file = os.path.join(scaler_dir, "scaler_y.save")
        logging.info(f"Loading target scaler from {scaler_y_file}")
        scaler_y = joblib.load(scaler_y_file)
        return scaler_y
    except Exception as e:
        logging.error(f"Failed to load target scaler: {e}")
        raise

def compute_metrics(predictions, y_true):
    """
    Compute evaluation metrics: MAE, MSE, MAPE, and R².
    """
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions) * 100
    r2 = r2_score(y_true, predictions)
    logging.info(f"Computed Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    return mae, mse, mape, r2


def save_evaluation_results(config, metrics, predictions, y_true):
    """
    Save evaluation results to a JSON file.
    """
    mae, mse, mape, r2 = metrics
    os.makedirs(config['evaluation_output_dir'], exist_ok=True)
    evaluation_results = {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "R2": r2,
        "Predictions": predictions.tolist(),
        "Actual": y_true.tolist()
    }
    evaluation_file = os.path.join(
        config['evaluation_output_dir'], "evaluation_results_with_sentiment.json")
    try:
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        logging.info(f"Evaluation results saved to {evaluation_file}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results: {e}")
        raise


def generate_and_save_plot(config, predictions, y_true):
    """
    Generate and save a plot comparing predictions and actual values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(predictions, label="Predicted", color="orange", linestyle="--")
    plt.title("Model Evaluation: Predictions vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(
        config['evaluation_output_dir'], "evaluation_plot_with_sentiment.png")
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Evaluation plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        raise


@log_time
def evaluate_model_with_sentiment(config):
    """
    Evaluate the trained LSTM model with sentiment features.
    """
    try:
        model, X_test, y_test = load_model_and_data(config)
        scaler_X = load_scaler_X(config)
        scaler_y = load_scaler_y(config)
        logging.info("Generating predictions...")
        predictions = model.predict(X_test)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        predictions_original = scaler_y.inverse_transform(predictions)
        y_test_original = scaler_y.inverse_transform(y_test)
        metrics = compute_metrics(predictions_original, y_test_original)
        save_evaluation_results(config, metrics, predictions_original, y_test_original)
        generate_and_save_plot(config, predictions_original, y_test_original)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        raise


def main(config_path: str = "config.json") -> None:
    """
    Main function to evaluate the trained model using the provided configuration.
    """
    setup_logging()
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        exit(1)
    evaluate_model_with_sentiment(config)


if __name__ == "__main__":
    main()

