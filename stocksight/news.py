# stocksight/news.py

import os
import json
import csv
import time
import logging
from datetime import datetime, timedelta
from typing import List

import requests
import pandas as pd
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import backoff
from ratelimit import limits, sleep_and_retry

from .utils import load_config

# Configuration Constants
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
QUERY = "business financial technology economics market"

MAX_RETRIES = 2
BACKOFF_FACTOR = 2
MAX_PAGES = 10
RATE_LIMIT_CALLS = 5 
RATE_LIMIT_PERIOD = 60

def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def is_valid_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def validate_start_date(start_date_str: str) -> datetime:
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        return start_date
    except ValueError:
        logging.error("start_date must be in 'YYYY-MM-DD' format.")
        raise

def load_dates_from_stock_data(price_file: str, start_date_dt: datetime) -> List[str]:
    try:
        stock_data = pd.read_csv(price_file)
        if 'Date' not in stock_data.columns:
            logging.error("Stock data CSV does not contain 'Date' column.")
            raise KeyError("Missing 'Date' column.")
        valid_dates = []
        for d in stock_data['Date'].dropna().unique():
            if is_valid_date(str(d)):
                dt_obj = datetime.strptime(str(d), "%Y-%m-%d")
                if dt_obj >= start_date_dt:
                    valid_dates.append(dt_obj)
        valid_dates.sort()
        return [dt.strftime("%Y-%m-%d") for dt in valid_dates]
    except Exception as e:
        logging.error(f"Failed to load dates from {price_file}: {e}")
        raise

def initialize_output_file(output_file: str) -> None:
    if not os.path.exists(output_file):
        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Articles'])
            logging.info(f"Created new output file with headers: {output_file}")
        except Exception as e:
            logging.error(f"Failed to initialize output file {output_file}: {e}")
            raise

def get_existing_dates(output_file: str) -> set:
    existing_dates = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_str = row.get('Date', '').strip()
                    if is_valid_date(date_str):
                        existing_dates.add(date_str)
            logging.info(f"Loaded {len(existing_dates)} existing dates from {output_file}.")
        except Exception as e:
            logging.error(f"Failed to read existing output file {output_file}: {e}")
    return existing_dates

def append_to_csv(output_file: str, date: str, articles: List[str]) -> None:
    articles_str = json.dumps(articles, ensure_ascii=False)
    try:
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([date, articles_str])
        logging.info(f"Appended {len(articles)} articles for date {date} -> {output_file}.")
    except Exception as e:
        logging.error(f"Failed to append data for date {date}: {e}")

@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
def make_api_request(session: requests.Session, params: dict) -> requests.Response:
    response = session.get(BASE_URL, params=params, timeout=10)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
        time.sleep(retry_after)
        raise Exception("Rate limit exceeded")
    response.raise_for_status()
    return response

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    jitter=backoff.full_jitter,
    giveup=lambda e: isinstance(e, requests.exceptions.HTTPError)
                    and e.response is not None
                    and e.response.status_code < 500
)
def fetch_articles_for_date(session: requests.Session, date: str, api_key: str, max_pages: int = MAX_PAGES) -> List[str]:
    articles = []
    # Calculate range: date to date+1 to ensure we get results (single day range often fails)
    dt = datetime.strptime(date, "%Y-%m-%d")
    next_dt = dt + timedelta(days=1)
    begin_str = dt.strftime("%Y%m%d")
    end_str = next_dt.strftime("%Y%m%d")
    
    for page in range(max_pages):
        params = {
            'begin_date': begin_str,
            'end_date': end_str,
            'api-key': api_key,
            'page': page,
            'sort': 'newest'
        }
        if QUERY:
            params['q'] = QUERY
            
        try:
            logging.info(f"Request for date {date} (range {begin_str}-{end_str}), page {page}.")
            response = make_api_request(session, params)
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            if not docs:
                logging.info(f"No more articles found for date {date} on page {page}.")
                break
                
            found_relevant_on_page = 0
            for doc in docs:
                # Client-side date check
                pub_date = doc.get('pub_date')
                if not pub_date or pub_date[:10] != date:
                    continue

                # Client-side filtering (Redundant but safe)
                doc_type = doc.get('document_type')
                desk = doc.get('news_desk', '')
                section = doc.get('section_name', '')
                
                # Broaden criteria slightly but avoid recipes/sports
                if doc_type == 'article':
                     # Check if relevant topic
                     relevant_topics = ["Business", "Financial", "Technology", "Economics", "Your Money", "DealBook", "Washington", "Foreign", "Politics"]
                     if (desk in relevant_topics) or (section in relevant_topics) or ("Business" in desk) or ("Business" in section):
                        headline = doc.get('headline', {}).get('main')
                        if headline:
                            articles.append(headline)
                            found_relevant_on_page += 1
                            
            if len(docs) < 10:
                logging.info(f"Less than 10 articles (raw) found for {date} page {page}, ending pagination.")
                break
        except Exception as e:
            logging.error(f"Request failed for date {date} page {page}: {e}")
            break
    return articles

def fetch_news_data(
    dates: List[str],
    output_file: str,
    api_key: str,
    max_pages_per_date: int = MAX_PAGES
) -> None:
    session = create_session()
    initialize_output_file(output_file)
    existing_dates = get_existing_dates(output_file)
    filtered_dates = [d for d in dates if d not in existing_dates]
    logging.info(f"Dates to process: {len(filtered_dates)} new dates.")
    if not filtered_dates:
        logging.info("No new dates to fetch.")
        return
    for date in tqdm(filtered_dates, desc="Fetching news data"):
        articles = fetch_articles_for_date(session, date, api_key, max_pages=max_pages_per_date)
        if articles:
            append_to_csv(output_file, date, articles)
        else:
            logging.warning(f"No articles found for date {date}.")
    logging.info(f"News fetching complete. Data appended in {output_file}.")
