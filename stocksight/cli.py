#!/usr/bin/env python
#================================
#  stocksight
#================================
"""
Orchestrates the entire pipeline:
    1) Fetch stock data (yfinance + TA-Lib)  [subcommand: stock]
    2) Fetch news data (NYT API)             [subcommand: news]
    3) Merge sentiment & create sequences    [subcommand: data]
    4) Train the model                       [subcommand: train]
    5) Evaluate the model                    [subcommand: evaluate]
    6) Predict future stock prices           [subcommand: predict]

Usage Examples:
    python main.py stock --ticker ^GSPC --start 2025-01-01 --end 2025-06-01
    python main.py news --start 2025-01-01
    python main.py data --config config.json --output data/merged_file.csv --cleaned /data/cleaned_file.csv --parallel
    python main.py train --config config.json
    python main.py evaluate --config config.json
    python main.py predict --config config.json --num-days 5
"""

import argparse
import logging
import sys
import os
import time
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, Any, List

import requests
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------------
# Global Logging Setup
# ------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# ------------------------------------------------------------------------
# CONFIGURATION LOADING
# ------------------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and return the configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Main script for entire pipeline: stock, news, data, train, evaluate, predict."
    )
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    #-------------------------------------------#
    # 1) STOCK (stock data) sub-command
    #-------------------------------------------#
    stock_parser = subparsers.add_parser('stock', help='Fetch stock data (yfinance + TA-Lib)')
    stock_parser.add_argument("-s", "--start", type=str, default="1985-01-01",
                              help="Start date (YYYY-MM-DD). Default=1985-01-01")
    stock_parser.add_argument("-i", "--interval", type=str, default="1d",
                              help="Data interval (e.g., 1d, 15m). Default=1d")
    stock_parser.add_argument("-c", "--config", type=str, default="config.json",
                              help="Path to config file. Default=config.json")
    stock_parser.add_argument("-e", "--end", type=str, default=None,
                              help="End date (YYYY-MM-DD). Default=today")
    stock_parser.add_argument("-t", "--ticker", type=str, default="^GSPC",
                              help="Ticker symbol (e.g. ^GSPC). Default=^GSPC")
    stock_parser.add_argument("-o", "--output", type=str, default=None,
                              help="Output CSV path. Default=data/stocks/<ticker>.csv")

    #-------------------------------------------#
    # 2) NEWS sub-command
    #-------------------------------------------#
    news_parser = subparsers.add_parser('news', help='Fetch news articles from the NYT API')
    news_parser.add_argument("-s", "--start", type=str, default="2024-12-20",
                             help="Start date (YYYY-MM-DD) for news fetch. Default=2024-12-20")

    #-------------------------------------------#
    # 3) DATA sub-command
    #-------------------------------------------#
    data_parser = subparsers.add_parser('data', help='Compute sentiments, merge data, create sequences')
    data_parser.add_argument("-c", "--config", type=str, default="config.json",
                              help="Path to the data config file.")
    data_parser.add_argument("-o", "--output", type=str, default=None,
                              help="Path to save the merged sentiments and price data.")
    data_parser.add_argument("-x", "--cleaned", type=str, default=None,
                              help="Path to save the cleaned CSV file for sentiments. If None, skip cleaning step.")
    data_parser.add_argument("-p", "--parallel", action='store_true',
                              help="Enable parallel processing for FinBERT sentiment analysis.")

    #-------------------------------------------------#
    # 4) TRAIN sub-command
    #-------------------------------------------------#
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument("-c", "--config", type=str, default="config.json",
                              help="Path to the training config file.")
    train_parser.add_argument("-g", "--grid", action='store_true',
                              help="Enable hyperparameter tuning (grid/random search).")

    #-------------------------------------------------#
    # 5) EVALUATE sub-command
    #-------------------------------------------------#
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument("-c", "--config", type=str, default="config.json",
                              help="Path to the evaluation config file.")
    eval_parser.add_argument("-m", "--metrics", type=str, default="mae,mse,mape,r2",
                              help="Comma-separated list of metrics to evaluate (e.g., mape,smape).")

    #-------------------------------------------------#
    # 6) PREDICT sub-command
    #-------------------------------------------------#
    predict_parser = subparsers.add_parser('predict', help='Predict future stock prices using the trained model')
    predict_parser.add_argument("-c", "--config", type=str, default="config.json",
                                 help="Path to the prediction config file.")
    predict_parser.add_argument("-n", "--num-days", type=int, default=5,
                                 help="Number of future days to predict. Default=5")

    #-------------------------------------------------#
    # Parse and dispatch
    #-------------------------------------------------#
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    #---------------- STOCK sub-command ----------------#
    if args.command == 'stock':
        from . import stock  # stock.py
        cfg = load_config(args.config)
        start_date = args.start
        interval = args.interval
        end_date = args.end
        ticker = args.ticker
        default_dir = cfg.get('data_dir', 'data')
        os.makedirs(default_dir, exist_ok=True)
        output_path = args.output if args.output else cfg['price_file']
        success = stock.fetch_price_data(ticker, start_date, end_date, interval, output_path)
        if success:
            logging.info("Stock data fetch & processing completed successfully.")
        else:
            logging.error("Failed to fetch or process stock data.")
        sys.exit(0)

    #---------------- NEWS sub-command ----------------#
    elif args.command == 'news':
        from . import news
        
        config = load_config("config.json")
        price_file = config['price_file']
        news_file = config['news_file']
        
        # Get API Key from config or env
        api_key = os.getenv("NYT_API_KEY") or config.get('nyt_api_key')
        if not api_key:
             logging.error("NYT_API_KEY not found in config.json or environment variables.")
             raise ValueError("Missing API Key")
             
        try:
            start_dt = news.validate_start_date(args.start)
            dates = news.load_dates_from_stock_data(price_file, start_dt)
            news.fetch_news_data(dates, news_file, api_key)
        except Exception as e:
            logging.error(f"Failed to execute news subcommand: {e}")
            sys.exit(1)
            
        sys.exit(0)

    #---------------- DATA sub-command ----------------#
    elif args.command == 'data':
        # GPU Configuration: Use tensorflow-directml if available (for Intel Xe Graphics)
        try:
            import tensorflow_directml_plugin as tf
            logging.info("Using tensorflow-directml for GPU acceleration.")
        except ImportError:
            import tensorflow as tf
            logging.info("Using standard TensorFlow.")


        def configure_intel_gpu():
            """
            Configure TensorFlow to use the Intel Xe Graphics GPU if available.
            If tensorflow-directml is used, GPU support is enabled automatically.
            Otherwise, try to enable memory growth on detected GPUs.
            """
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        details = tf.config.experimental.get_device_details(gpu)
                        device_name = details.get('device_name', gpu.name)
                        if "Intel" in device_name or "Xe" in device_name:
                            logging.info(f"Using Intel GPU: {device_name}")
                            try:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            except RuntimeError as e:
                                logging.error(f"Error setting memory growth on GPU: {e}")
                        else:
                            logging.info(f"Found GPU but not Intel Xe: {device_name}")
                else:
                    logging.info("No GPU found. Using CPU.")
            except Exception as e:
                logging.error(f"Error during GPU configuration: {e}")


        # Configure Intel Xe Graphics GPU if available
        configure_intel_gpu()

        from . import data  # data.py in the current directory
        logging.info("Starting sentiment computation step (FinBERT) before data setup...")
        cfg = load_config(args.config)
        
        if os.path.exists(cfg['sentiment_file']):
            logging.info(f"Sentiment file '{cfg['sentiment_file']}' already exists. Skipping sentiment computation.")
        else:
            try:
                from . import sentiments  # sentiments.py should handle FinBERT sentiment analysis
                sentiments.compute_sentiment(
                    input_file=cfg['news_file'],
                    output_file=cfg['sentiment_file'],
                    cleaned_file=args.cleaned,
                    batch_size=cfg.get('batch_size', 32),
                    parallel=args.parallel
                )
                logging.info("Sentiment computation completed successfully.")
            except Exception as e:
                logging.error(f"Sentiment computation failed: {e}")
                sys.exit(1)
                
        logging.info("Proceeding with standard data setup (merging, sequences)...")
        data.main(config_path=args.config)
        sys.exit(0)

    #---------------- TRAIN sub-command ----------------#
    elif args.command == 'train':
        # GPU Configuration: Use tensorflow-directml if available (for Intel Xe Graphics)
        try:
            import tensorflow_intel as tf
            logging.info("Using tensorflow-intel for GPU acceleration.")
        except ImportError:
            import tensorflow as tf
            logging.info("Using standard TensorFlow.")


        def configure_intel_gpu():
            """
            Configure TensorFlow to use the Intel Xe Graphics GPU if available.
            If tensorflow-directml is used, GPU support is enabled automatically.
            Otherwise, try to enable memory growth on detected GPUs.
            """
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        details = tf.config.experimental.get_device_details(gpu)
                        device_name = details.get('device_name', gpu.name)
                        if "Intel" in device_name or "Xe" in device_name:
                            logging.info(f"Using Intel GPU: {device_name}")
                            try:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            except RuntimeError as e:
                                logging.error(f"Error setting memory growth on GPU: {e}")
                        else:
                            logging.info(f"Found GPU but not Intel Xe: {device_name}")
                else:
                    logging.info("No GPU found. Using CPU.")
            except Exception as e:
                logging.error(f"Error during GPU configuration: {e}")


        # Configure Intel Xe Graphics GPU if available
        configure_intel_gpu()

        from . import train_model as train
        train.main(args.config, use_tuning=args.grid)
        sys.exit(0)

    #---------------- EVALUATE sub-command ----------------#
    elif args.command == 'evaluate':
        from . import evaluate_model as evaluate
        evaluate.main(config_path=args.config, metrics_str=args.metrics)
        sys.exit(0)

    #---------------- PREDICT sub-command ----------------#
    elif args.command == 'predict':
        from . import predict_future_price as predict
        predict.main(config_path=args.config, num_days=args.num_days)
        sys.exit(0)


if __name__ == "__main__":
    main()
