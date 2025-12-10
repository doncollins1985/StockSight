import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stocksight.train_model import train_one_epoch, validate, calculate_ctm_loss
from stocksight.models import StockPredictor

class TestTrainingLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.input_shape = (10, 3) # seq_len=10, features=3
        self.hp = {
            'd_model': 16,
            'memory_length': 5,
            'n_ticks': 3,
            'j_out': 8,
            'j_action': 8,
            'n_heads': 2,
            'nlm_hidden': 8
        }
        self.model = StockPredictor(self.input_shape, self.hp).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Create dummy data
        # batch_size=4, seq_len=10, features=3
        self.inputs = torch.randn(4, 10, 3).to(self.device)
        self.targets = torch.randn(4).to(self.device)
        
        dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(dataset, batch_size=2)

    def test_train_one_epoch_runs(self):
        """Test if train_one_epoch runs without errors with the new CTM model and loss."""
        try:
            loss = train_one_epoch(self.model, self.dataloader, None, self.optimizer, self.device, direction_weight=0.5)
            print(f"Training loss: {loss}")
            self.assertIsInstance(loss, float)
        except Exception as e:
            self.fail(f"train_one_epoch raised an exception: {e}")
            
    def test_validate_runs(self):
        """Test if validate runs without errors."""
        try:
            loss = validate(self.model, self.dataloader, None, self.device, direction_weight=0.5)
            print(f"Validation loss: {loss}")
            self.assertIsInstance(loss, float)
        except Exception as e:
            self.fail(f"validate raised an exception: {e}")

    def test_loss_calculation_logic(self):
        """Verify the CTM loss calculation logic."""
        outputs = torch.tensor([[[1.0], [1.5], [2.0]]]) # B=1, T=3, Preds approaching target
        sigmas = torch.tensor([[[1.0], [0.5], [0.1]]]) # B=1, T=3, Certainty increasing (sigma decreasing)
        targets = torch.tensor([2.0]) # Target=2.0
        inputs = torch.zeros(1, 10, 3) # Dummy inputs
        inputs[0, -1, 0] = 1.0 # Current Price (for directional penalty)
        
        # calculate_ctm_loss
        loss = calculate_ctm_loss(outputs, sigmas, targets, inputs, direction_weight=0.0)
        
        # Expected Logic:
        # Tick 1: pred=1.0, sigma=1.0. NLL ~ 0.5*log(1) + 0.5*(2-1)^2/1 = 0 + 0.5 = 0.5
        # Tick 2: pred=1.5, sigma=0.5. NLL ~ 0.5*log(0.25) + 0.5*(2-1.5)^2/0.25 = -0.69 + 0.5*0.25*4 = -0.69 + 0.5 = -0.19
        # Tick 3: pred=2.0, sigma=0.1. NLL ~ 0.5*log(0.01) + 0.5*(0)/0.01 = -2.3 + 0 = -2.3
        
        # Min Loss = -2.3 (Tick 3)
        # Min Sigma = 0.1 (Tick 3) -> Loss at Min Sigma = -2.3
        
        # Loss = (-2.3 + -2.3) / 2 = -2.3
        
        self.assertAlmostEqual(loss.item(), -2.302585, places=2)

if __name__ == '__main__':
    unittest.main()