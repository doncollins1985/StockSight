import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stocksight import news

class TestNewsCollection(unittest.TestCase):

    def test_validate_start_date_valid(self):
        """Test valid date string."""
        date_str = "2023-01-01"
        dt = news.validate_start_date(date_str)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.strftime("%Y-%m-%d"), date_str)

    def test_validate_start_date_invalid(self):
        """Test invalid date string."""
        date_str = "invalid-date"
        with self.assertRaises(ValueError):
            news.validate_start_date(date_str)

    @patch('stocksight.news.pd.read_csv')
    def test_load_dates_from_stock_data(self, mock_read_csv):
        """Test loading dates from stock data CSV."""
        # Mock stock data
        mock_df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', 'invalid', '2022-12-31']
        })
        mock_read_csv.return_value = mock_df

        start_date = datetime(2023, 1, 1)
        dates = news.load_dates_from_stock_data("dummy.csv", start_date)
        
        # Expected dates: 2023-01-01, 2023-01-02, 2023-01-03
        # 2022-12-31 is before start_date
        self.assertEqual(len(dates), 3)
        self.assertIn("2023-01-01", dates)
        self.assertIn("2023-01-02", dates)
        self.assertIn("2023-01-03", dates)
        self.assertNotIn("2022-12-31", dates)

    @patch('stocksight.news.create_session')
    @patch('stocksight.news.append_to_csv')
    @patch('stocksight.news.get_existing_dates')
    @patch('stocksight.news.initialize_output_file')
    def test_fetch_news_data(self, mock_init, mock_get_existing, mock_append, mock_create_session):
        """Test fetching news data."""
        # Mock session and response
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': {
                'docs': [
                    {
                        'pub_date': '2023-01-01T10:00:00+0000',
                        'document_type': 'article',
                        'news_desk': 'Business',
                        'headline': {'main': 'Test Article 1'}
                    }
                ]
            }
        }
        mock_session.get.return_value = mock_response
        mock_create_session.return_value = mock_session

        # Mock existing dates (none)
        mock_get_existing.return_value = set()

        dates = ['2023-01-01']
        api_key = "dummy_key"
        output_file = "test_news.csv"

        news.fetch_news_data(dates, output_file, api_key, max_pages_per_date=1)

        # Verify calls
        mock_init.assert_called_once_with(output_file)
        mock_create_session.assert_called_once()
        # Should call append_to_csv once for the date
        mock_append.assert_called_once()
        args, _ = mock_append.call_args
        self.assertEqual(args[0], output_file)
        self.assertEqual(args[1], '2023-01-01')
        self.assertEqual(len(args[2]), 1)
        self.assertEqual(args[2][0], 'Test Article 1')

    @patch('stocksight.news.create_session')
    @patch('stocksight.news.append_to_csv')
    @patch('stocksight.news.get_existing_dates')
    @patch('stocksight.news.initialize_output_file')
    def test_fetch_news_data_existing(self, mock_init, mock_get_existing, mock_append, mock_create_session):
        """Test fetching news data when date already exists."""
        mock_get_existing.return_value = {'2023-01-01'}
        
        dates = ['2023-01-01']
        api_key = "dummy_key"
        output_file = "test_news.csv"

        news.fetch_news_data(dates, output_file, api_key)

        # Should NOT call append_to_csv because date exists
        mock_append.assert_not_called()

if __name__ == '__main__':
    unittest.main()
