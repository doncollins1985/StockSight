#!/usr/bin/env python3
# compute_sentiments.py

import os
import csv
import json
import ast
import logging
import logging.handlers
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
from multiprocessing import cpu_count, Pool, Queue, Process, current_process
from .utils import load_config
import sys
import time
import colorlog  # Import colorlog for colored logging
import torch

# Global variable to hold the NLP model in each worker
nlp = None

def listener_configurer(log_file='logs/compute_sentiments.log'):
    """
    Configures the listener logger to write log records to a file and the console with colors.
    
    Args:
        log_file (str): Path to the log file.
    """
    root = logging.getLogger()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        },
        secondary_log_colors={},
        style='%'
    )
    
    # File Handler (No color)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    # Console Handler (With color)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    root.addHandler(file_handler)
    root.addHandler(console_handler)

def listener_process(queue, log_file):
    """
    The listener process that receives log records from the queue and logs them.
    
    Args:
        queue (Queue): Multiprocessing queue for log records.
        log_file (str): Path to the log file.
    """
    listener_configurer(log_file)
    while True:
        try:
            record = queue.get()
            if record is None:
                # Sentinel received, terminate listener
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys
            import traceback
            print('Problem handling log record:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def load_model(batch_size=32):
    """
    Load the pre-trained FinBERT model and tokenizer.

    Args:
        batch_size (int): Number of samples per batch for sentiment analysis.

    Returns:
        pipeline: Hugging Face transformers pipeline for sentiment analysis.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading FinBERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    device = 0 if torch.cuda.is_available() else -1
    nlp_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    logger.info(f"Model and tokenizer loaded successfully on device {device}.")
    return nlp_pipeline


def initializer(batch_size, queue):
    """
    Initializer function for each worker in the multiprocessing pool.
    Sets up the logger to use the logging queue and loads the NLP model.

    Args:
        batch_size (int): Number of samples per batch for sentiment analysis.
        queue (Queue): Multiprocessing queue for log records.
    """
    global nlp
    nlp = load_model(batch_size)

def clean_csv(input_file, cleaned_file, queue):
    """
    Clean the CSV by ensuring proper quoting and removing malformed lines.

    Args:
        input_file (str): Path to the original CSV file.
        cleaned_file (str): Path to save the cleaned CSV file.
        queue (Queue): Multiprocessing queue for log records.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CSV cleaning: {input_file} -> {cleaned_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(cleaned_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for i, row in enumerate(reader, start=2):  # Start at 2 considering header
            if len(row) >= 2:
                writer.writerow(row)
            else:
                logger.warning(f"Skipping malformed line {i}: {row}")
    logger.info("CSV cleaning completed.")

def parse_articles(articles_str, queue=None):
    """
    Parse the 'Articles' string from the CSV into a list.
    Attempts JSON parsing first, then falls back to ast.literal_eval.

    Args:
        articles_str (str): String representation of articles.
        queue (Queue, optional): Multiprocessing queue for log records.

    Returns:
        list: List of articles.
    """
    logger = logging.getLogger(__name__)
    if pd.isna(articles_str):
        return []
    try:
        # Attempt to parse as JSON
        return json.loads(articles_str)
    except json.JSONDecodeError:
        try:
            # Fallback to literal_eval
            return ast.literal_eval(articles_str)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error parsing Articles: {e} | Data: {articles_str}")
            return []

def analyze_sentiments(articles):
    """
    Perform sentiment analysis on a list of articles.
    Returns the proportion of positive, negative, and neutral sentiments,
    along with an aggregate score.

    Args:
        articles (list): List of article texts.

    Returns:
        tuple: (positive_ratio, negative_ratio, neutral_ratio, aggregate_score)
    """
    logger = logging.getLogger(__name__)
    if not articles:
        return 0.0, 0.0, 0.0, 0.0
    try:
        results = nlp(articles)
        labels = [res['label'].lower() for res in results]
        pos = labels.count('positive') / len(labels)
        neg = labels.count('negative') / len(labels)
        neu = labels.count('neutral') / len(labels)
        aggregate_score = pos - neg  # Example: Positive minus Negative
        return pos, neg, neu, aggregate_score
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return 0.0, 0.0, 0.0, 0.0

def process_row(row):
    """
    Process a single row of data to compute sentiment.
    Assumes that the NLP model has been loaded globally.

    Args:
        row (dict): A dictionary representing a row from the grouped DataFrame.

    Returns:
        dict: Sentiment analysis results for the row.
    """
    logger = logging.getLogger(__name__)
    try:
        date = row['Date']
        articles = row['Parsed_Articles']
        pos, neg, neu, agg = analyze_sentiments(articles)
        return {
            'Date': date,
            'Positive': pos,
            'Negative': neg,
            'Neutral': neu,
            'Aggregate_Score': agg
        }
    except Exception as e:
        logger.error(f"Error processing row for date {row.get('Date', 'Unknown')}: {e}")
        return {
            'Date': row.get('Date', 'Unknown'),
            'Positive': None,
            'Negative': None,
            'Neutral': None,
            'Aggregate_Score': None
        }

def remove_duplicates(news_data, queue):
    """
    Remove duplicate articles based on 'Date' and 'Articles'.

    Args:
        news_data (DataFrame): Pandas DataFrame containing news data.
        queue (Queue): Multiprocessing queue for log records.

    Returns:
        DataFrame: Deduplicated DataFrame.
    """
    logger = logging.getLogger(__name__)
    initial_count = news_data.shape[0]
    news_data = news_data.drop_duplicates(subset=['Date', 'Articles'])
    final_count = news_data.shape[0]
    logger.info(f"Removed {initial_count - final_count} duplicate rows.")
    return news_data

def handle_missing_values(news_data, queue):
    """
    Handle missing values in the DataFrame by removing rows with missing 'Date' or 'Articles'.

    Args:
        news_data (DataFrame): Pandas DataFrame containing news data.
        queue (Queue): Multiprocessing queue for log records.

    Returns:
        DataFrame: Cleaned DataFrame with no missing 'Date' or 'Articles'.
    """
    logger = logging.getLogger(__name__)
    initial_count = news_data.shape[0]
    news_data = news_data.dropna(subset=['Date', 'Articles'])
    final_count = news_data.shape[0]
    logger.info(f"Removed {initial_count - final_count} rows with missing 'Date' or 'Articles'.")
    return news_data

def compute_sentiment(input_file, output_file, cleaned_file=None, batch_size=32, parallel=False, queue=None):
    """
    Main function to compute sentiment scores per date.
    Aggregates all articles for each date before analysis.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output sentiment CSV.
        cleaned_file (str, optional): Path to save the cleaned CSV. Defaults to None.
        batch_size (int, optional): Batch size for sentiment analysis. Defaults to 32.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
        queue (Queue, optional): Multiprocessing queue for log records.
    """ 
    logger = logging.getLogger(__name__)
    # Step 1: Clean the CSV if a cleaned_file path is provided
    if cleaned_file:
        clean_csv(input_file, cleaned_file, queue)
        data_file = cleaned_file
    else:
        data_file = input_file

    # Step 2: Read the CSV with appropriate parameters
    try:
        news_data = pd.read_csv(
            data_file,
            engine='python',
            delimiter=',',
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\',
            on_bad_lines='warn',
            encoding='utf-8'
        )
        logger.info(f"Successfully read input file: {data_file}")
    except pd.errors.ParserError as e:
        logger.error(f"ParserError: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error reading CSV: {e}")
        return

    logger.info(f"Columns in CSV: {news_data.columns.tolist()}")

    # Step 3: Validate required columns
    required_columns = {'Date', 'Articles'}
    if not required_columns.issubset(news_data.columns):
        missing = required_columns - set(news_data.columns)
        logger.error(f"Input file is missing required columns: {missing}")
        return

    # Step 4: Data Cleaning
    news_data = remove_duplicates(news_data, queue)
    news_data = handle_missing_values(news_data, queue)

    # Step 5: Parse the 'Articles' column into lists
    news_data['Parsed_Articles'] = news_data['Articles'].apply(parse_articles)

    # Step 6: Group by 'Date' and aggregate all articles for each date
    grouped = news_data.groupby('Date')['Parsed_Articles'].apply(
        lambda lists: [article for sublist in lists for article in sublist]
    ).reset_index()

    logger.info(f"Aggregated articles by date. Total unique dates: {grouped.shape[0]}")

    # Optional: Log a sample of the aggregated data
    if not grouped.empty:
        sample_date = grouped['Date'].iloc[0]
        sample_articles = grouped['Parsed_Articles'].iloc[0]
        logger.info(f"Sample 'Articles' for {sample_date}: {sample_articles[:2]}{'...' if len(sample_articles) > 2 else ''}")

    # Step 7: Perform Sentiment Analysis
    sentiments = []

    if parallel:
        logger.info("Starting sentiment analysis with parallel processing...")
        try:
            with Pool(
                processes=cpu_count(),
                initializer=initializer,
                initargs=(batch_size, queue)
            ) as pool:
                # Convert DataFrame rows to dictionaries for easier handling
                rows = grouped.to_dict('records')
                for sentiment in tqdm(pool.imap(process_row, rows), total=grouped.shape[0], desc="Processing dates"):
                    sentiments.append(sentiment)
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
    else:
        logger.info("Starting sentiment analysis with sequential processing...")
        global nlp
        nlp = load_model(batch_size)
        for row in tqdm(grouped.to_dict('records'), total=grouped.shape[0], desc="Processing dates"):
            sentiment = process_row(row)
            sentiments.append(sentiment)

    # Step 8: Create DataFrame and Save Results
    sentiment_df = pd.DataFrame(sentiments)

    # Optional: Reorder columns
    sentiment_df = sentiment_df[['Date', 'Positive', 'Negative', 'Neutral', 'Aggregate_Score']]

    # Save to CSV
    try:
        sentiment_df.to_csv(output_file, index=False)
        logger.info(f"Sentiment data saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving sentiment data to CSV: {e}")

def main():
    """
    Entry point of the script. Parses command-line arguments and initiates sentiment computation.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Compute sentiment scores from news articles CSV.")
    parser.add_argument('-x', '--cleaned', type=str, default=None, help="Path to save the cleaned CSV file. If not provided, cleaning is skipped.")
    parser.add_argument('-p', '--parallel', action='store_true', help="Enable parallel processing for sentiment analysis.")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config('config.json')

    # Setup logging queue
    log_queue = Queue()
    log_listener = Process(target=listener_process, args=(log_queue, 'compute_sentiments.log'))
    log_listener.start()

    # Configure the main process logger to use the queue
    try:
        compute_sentiment(
            input_file=config['news_file'],
            output_file=config['sentiment_file'],
            cleaned_file=args.cleaned,
            batch_size=config['batch_size'],
            parallel=args.parallel,
            queue=log_queue
        )
    finally:
        # Send sentinel to stop the listener
        log_queue.put_nowait(None)
        log_listener.join()

if __name__ == "__main__":
    main()

