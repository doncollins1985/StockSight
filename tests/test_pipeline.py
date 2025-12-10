import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stocksight import stock, data, train_model
from stocksight.models import StockPredictor

class TestStockSightPipeline(unittest.TestCase):

    def setUp(self):
        self.config = {
            "price_file": "test_price.csv",
            "sentiment_file": "test_sentiments.csv",
            "merged_file": "test_merged.csv",
            "sequence_file": "test_sequences.npz",
            "feature_columns": ["Close", "SMA_5", "Positive"],
            "window_size": 5,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "scaler_file_X": "scaler_X.save",
            "scaler_file_y": "scaler_y.save",
            "batch_size": 2,
            "epochs": 1,
            "early_stopping_patience": 1,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 1,
            "reduce_lr_min_lr": 1e-6,
            "warmup_epochs": 0,
            "tensorboard_log_dir": "logs/",
            "checkpoint_dir": "checkpoints/",
            "output_model_file": "models/final_model.pth",
            "history_file": "history.json"
        }

    @patch('stocksight.stock.yf.Ticker')
    def test_stock_fetch(self, mock_ticker):
        # Mock yfinance data
        mock_hist = pd.DataFrame({
            'Open': [100.0] * 20,
            'High': [105.0] * 20,
            'Low': [95.0] * 20,
            'Close': [100.0 + i for i in range(20)],
            'Volume': [1000] * 20,
        })
        mock_hist.index = pd.date_range(start='2023-01-01', periods=20)
        mock_hist.index.name = 'Date'
        
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_hist
        mock_ticker.return_value = mock_stock

        # Test fetch_price_data
        with patch('stocksight.stock.logger') as mock_logger:
            success = stock.fetch_price_data('AAPL', '2023-01-01', '2023-01-20', '1d', 'test_price.csv')
            self.assertTrue(success)
            self.assertTrue(os.path.exists('test_price.csv'))

    def test_data_merge_and_sequence(self):
        # Create dummy price and sentiment data
        dates = pd.date_range(start='2023-01-01', periods=20)
        price_df = pd.DataFrame({
            'Date': dates,
            'Close': [100.0 + i for i in range(20)],
            'SMA_5': [100.0] * 20 # Dummy
        })
        price_df.to_csv('test_price.csv', index=False)

        sentiment_df = pd.DataFrame({
            'Date': dates,
            'Positive': np.random.rand(20),
            'Negative': np.random.rand(20),
            'Neutral': np.random.rand(20),
            'Aggregate_Score': np.random.rand(20)
        })
        sentiment_df.to_csv('test_sentiments.csv', index=False)

        # Mock load_config to return our test config
        with patch('stocksight.data.load_config', return_value=self.config):
             # Run main data pipeline steps directly
             data.merge_sentiment_with_stock(
                 self.config['price_file'],
                 self.config['sentiment_file'],
                 self.config['merged_file']
             )
             
             self.assertTrue(os.path.exists('test_merged.csv'))

             # Manually update config feature_columns for the test since we are not using the full set
             feature_cols = self.config['feature_columns']
             
             data.create_sequences_with_sentiment(
                 self.config['merged_file'],
                 self.config['sequence_file'],
                 feature_cols,
                 self.config['window_size']
             )
             
             self.assertTrue(os.path.exists('test_sequences.npz'))

    def test_model_build(self):
        # Test if PyTorch model (CTM) builds
        hp = {
            'd_model': 32,
            'memory_length': 5,
            'n_ticks': 3,
            'n_heads': 2,
            'nlm_hidden': 16,
            'j_out': 16,
            'j_action': 16
        }
        input_shape = (5, 3) # seq_len, features
        model = StockPredictor(input_shape, hp)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test forward pass
        dummy_input = torch.randn(2, 5, 3) # batch, seq, features
        output = model(dummy_input)
        # Default forward returns [B, 1]? Actually [B, d_model] -> No, W_out projects to 2 (mean, logsigma), but forward returns output[:, -1, :].
        # Wait, forward returns outputs[:, -1, :]. 
        # w_out output is (B, 2). 
        # So outputs is (B, T, 1). 
        # Let's check models.py again. w_out output is (B, 2) [mean, logsigma].
        # mean = preds[:, 0:1]. 
        # outputs_sequence.append(mean).
        # So outputs is (B, T, 1).
        # Return outputs[:, -1, :] -> (B, 1).
        
        self.assertEqual(output.shape, (2, 1))

    def tearDown(self):
        # Cleanup
        files = ['test_price.csv', 'test_sentiments.csv', 'test_merged.csv', 'test_sequences.npz', 'scaler_X.save', 'scaler_y.save']
        for f in files:
            if os.path.exists(f):
                os.remove(f)

if __name__ == '__main__':
    unittest.main()