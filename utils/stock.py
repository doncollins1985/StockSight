#!/usr/bin/env python
#====================================
#  stock.py
#====================================
"""
Script to fetch stock data using yfinance, compute technical indicators
using TA-Lib, and save the resulting CSV.
"""

import os
import logging
import argparse
import talib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

from utils.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a suite of technical indicators using TA-Lib and append them to df.
    Expects columns: ['Date', 'High', 'Low', 'Close', 'Volume'].

    Returns:
        DataFrame with new indicator columns appended.
    """
    required = {'High', 'Low', 'Close', 'Volume'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.error(f"DataFrame is missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    if 'Date' not in df.columns:
        logger.error("DataFrame is missing 'Date' column.")
        raise ValueError("Missing 'Date' column.")

    # Convert to float for TA-Lib compatibility
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)

    try:
        # Simple/Exponential MAs
        df['SMA_5'] = talib.SMA(close, timeperiod=5)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist

        # RSI
        df['RSI'] = talib.RSI(close, timeperiod=14)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # ATR
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        df['STOCH_K'] = stoch_k
        df['STOCH_D'] = stoch_d

        # OBV
        df['OBV'] = talib.OBV(close, volume)

        # Rate of Change
        df['ROC'] = talib.ROC(close, timeperiod=10)

        # ADX
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        # CCI
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        # Williams %R
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        # Momentum
        df['MOM'] = talib.MOM(close, timeperiod=10)

        # Price & Volume change
        df['Price_Change'] = close.pct_change()
        df['Volume_Change'] = volume.pct_change()

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise

    # Check if essential indicators exist
    required_columns = [
        'Date', 'Close', 'SMA_5', 'SMA_20', 'SMA_50', 'EMA_12',
        'EMA_26', 'MACD', 'MACD_Signal', 'RSI', 'BB_Middle',
        'ATR', 'STOCH_K', 'STOCH_D', 'OBV', 'ROC', 'ADX', 'CCI',
        'WILLR', 'MOM'
    ]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"After calculations, missing columns: {missing_cols}")
        raise ValueError(f"Missing columns after calculations: {missing_cols}")

    return df[required_columns]


def fetch_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    output_file: str
) -> bool:
    """
    Fetch price data for a given ticker from Yahoo Finance, compute indicators, save to CSV.
    """
    try:
        logger.info(
            f"Fetching {ticker} data from {start_date} to {end_date} with interval '{interval}'."
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Download data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.error(f"No data found for {ticker}.")
            return False

        # Reset index so 'Date' is a column
        data.reset_index(inplace=True)

        # Rename 'Datetime' to 'Date' if needed
        if 'Date' not in data.columns and 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
            logger.info("Renamed 'Datetime' column to 'Date'.")

        if 'Date' not in data.columns:
            logger.error("DataFrame has no 'Date' or 'Datetime' columns.")
            return False

        # Calculate TA indicators
        data = calculate_technical_indicators(data)

        # Fill NaNs
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        # Convert date format depending on interval
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            if interval.endswith(('m', 'h')):  # intraday
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M')
            else:
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        # Save to CSV
        data.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")

        # Log summary
        logger.info(f"Total rows: {len(data)}")
        logger.info(f"Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")
        logger.info(f"Close price range: {data['Close'].min():.2f} to {data['Close'].max():.2f}")
        logger.info(f"Columns included: {', '.join(data.columns)}")
        return True

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch and process stock data.")
    parser.add_argument(
        "-s", "--start", type=str, default="1985-01-01",
        help="Start date (YYYY-MM-DD). Default: 1985-01-01"
    )
    parser.add_argument(
        "-i", "--interval", type=str, default="1d",
        help="Data interval. Default: 1d (daily)."
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config.json",
        help="Path to the config file. Default: config.json"
    )
    parser.add_argument(
        "-e", "--end", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: current date"
    )
    parser.add_argument(
        "-t", "--ticker", type=str, default="^GSPC",
        help="Ticker symbol. Default: ^GSPC"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output CSV file path. Default: data/<ticker>.csv"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    start_date = args.start
    interval = args.interval
    end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")
    ticker = args.ticker

    default_dir = config.get('data_dir', 'data')
    os.makedirs(default_dir, exist_ok=True)
    output_path = args.output if args.output else config['stock_file']

    success = fetch_price_data(ticker, start_date, end_date, interval, output_path)
    if success:
        logger.info("Data collection and processing completed successfully.")
    else:
        logger.error("Failed to collect and process data.")


if __name__ == "__main__":
    main()

