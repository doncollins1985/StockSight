#!/usr/bin/env python
#====================================
#  stock.py
#====================================

import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
from utils import load_config
from datetime import datetime
import talib
from typing import Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a suite of technical indicators using TA-Lib and append them to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'High', 'Low', 'Close', 'Volume' columns.

    Returns:
        pd.DataFrame: DataFrame including calculated technical indicators.
    """
    required = {'High', 'Low', 'Close', 'Volume'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.error(f"DataFrame is missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure 'Date' is present
    if 'Date' not in df.columns:
        logger.error("DataFrame is missing 'Date' column.")
        raise ValueError("Missing 'Date' column.")

    # Ensure columns are float type for TA-Lib compatibility
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)

    try:
        # Moving averages
        df['SMA_5'] = talib.SMA(close, timeperiod=5)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)

        # Exponential moving averages
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

        # Average True Range
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        df['STOCH_K'] = stoch_k
        df['STOCH_D'] = stoch_d

        # On-Balance Volume
        df['OBV'] = talib.OBV(close, volume)

        # Rate of Change
        df['ROC'] = talib.ROC(close, timeperiod=10)

        # Average Directional Movement Index
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        # Commodity Channel Index
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        # Williams %R
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        # Momentum
        df['MOM'] = talib.MOM(close, timeperiod=10)

        # Percentage changes
        df['Price_Change'] = close.pct_change()
        df['Volume_Change'] = volume.pct_change()

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise

    required_columns = [
        'Date', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'ATR', 'STOCH_K', 'STOCH_D', 'OBV', 'ROC', 'ADX', 'CCI', 'WILLR', 'MOM',
        'Price_Change', 'Volume_Change'
    ]

    # Check if all required columns are present after calculations
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"After calculations, DataFrame is missing columns: {missing_cols}")
        raise ValueError(f"Missing columns after calculations: {missing_cols}")

    return df[required_columns]


def fetch_stock_data(ticker: str,
                     start_date: str,
                     end_date: str,
                     interval: str,
                     output_file: str) -> bool:
    """
    Fetch price data for a given ticker within a date range, calculate technical indicators, and save to CSV.

    Args:
        ticker (str): The ticker symbol (e.g. '^GSPC').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): The data interval (default '1d').
        output_file (str): The filepath to save the output CSV.

    Returns:
        bool: True if data was successfully fetched and saved, False otherwise.
    """
    try:
        logger.info(
            f"Fetching {ticker} data from {start_date} to {end_date} with interval '{interval}'."
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Download data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(
            start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.error(f"No data found for {ticker}.")
            return False

        # Reset index to ensure datetime is a column
        data.reset_index(inplace=True)

        # Log the columns for debugging
        logger.debug(f"Data columns after reset_index: {data.columns.tolist()}")

        # Rename 'Datetime' to 'Date' if necessary
        if 'Date' not in data.columns and 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
            logger.info("Renamed 'Datetime' column to 'Date'.")
        elif 'Date' not in data.columns and 'Datetime' not in data.columns:
            logger.error("DataFrame does not contain 'Date' or 'Datetime' columns.")
            return False

        # Log the first few rows for debugging
        logger.debug(f"Fetched data head:\n{data.head()}")

        # Calculate technical indicators
        data = calculate_technical_indicators(data)

        # Handle NaN values: forward fill then backfill
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        # Convert dates to string format and extract HH:MM
        if 'Date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Date']):
          if interval.endswith(('m', 'h')):
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M')
          else:
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        
        # Add Name Column at the beginning
        data.insert(0, 'Name', ticker)

        # Save to CSV
        data.to_csv(output_file, index=False)
        logger.info(f"Data successfully saved to {output_file}")

        # Log summary statistics
        logger.info(f"Total rows: {len(data)}")
        logger.info(
            f"Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}"
        )
        logger.info(
            f"Close price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}"
        )
        logger.info(f"Columns included: {', '.join(data.columns)}")

        return True

    except Exception as e:
        logger.error(f"Error fetching and processing data for {ticker}: {e}")
        return False


def main() -> None:
    """
    Main function to execute data fetching and processing for a given stock ticker.
    """
    parser = argparse.ArgumentParser(description="Fetch and process stock data.")
    parser.add_argument("-s", "--start", type=str, default="1985-01-01", help="Start date in '%Y-%m-%d' format (default: 1985-01-01)")
    parser.add_argument("-i", "--interval", type=str, default="1d", help="Data interval (default: 1d)")
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to the config file (default: config.json)")
    parser.add_argument("-e", "--end", type=str, default=None, help="End date in '%Y-%m-%d' format (default: current date)")
    parser.add_argument("-t", "--ticker", type=str, default="^GSPC", help="Stock ticker (default: ^GSPC)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV file path (default: data/<ticker>.csv)")

    args = parser.parse_args()

    config = load_config(args.config)

    # Use command line arguments if provided, otherwise use config file or defaults
    start_date = args.start
    interval = args.interval
    end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")
    ticker = args.ticker
    output_path = args.output if args.output else os.path.join(config.get('data_dir', 'data/stocks'), f'{ticker}.csv')

    # Check TA-Lib availability
    try:
        import talib  # noqa: F401
    except ImportError:
        logger.error("TA-Lib not installed. Please install TA-Lib first.")
        logger.error(
            "Installation instructions: https://github.com/mrjbq7/ta-lib"
        )
        return

    success = fetch_stock_data(ticker, start_date, end_date, interval, output_path)
    if success:
        logger.info("Data collection and processing completed successfully.")
    else:
        logger.error("Failed to collect and process data.")


if __name__ == "__main__":
    main()
