
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NLM(nn.Module):
    """
    Neuron-Level Model: Applies a private MLP to the history of pre-activations for each neuron.
    Implemented efficiently using einsum.
    """
    def __init__(self, d_model, memory_length, hidden_size):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.hidden_size = hidden_size
        
        # We want to implement D separate MLPs.
        # Layer 1: M -> hidden_size (per neuron)
        # Weights: (d_model, memory_length, hidden_size)
        self.w1 = nn.Parameter(torch.randn(d_model, memory_length, hidden_size) / math.sqrt(memory_length))
        self.b1 = nn.Parameter(torch.zeros(d_model, hidden_size))
        
        # Layer 2: hidden_size -> 1 (per neuron)
        # Weights: (d_model, hidden_size, 1)
        self.w2 = nn.Parameter(torch.randn(d_model, hidden_size, 1) / math.sqrt(hidden_size))
        self.b2 = nn.Parameter(torch.zeros(d_model, 1))
        
    def forward(self, pre_acts_history):
        # pre_acts_history: [Batch, d_model, memory_length]
        
        # Layer 1
        # Input: (B, D, M)
        # Weights: (D, M, H)
        # Output: (B, D, H)
        # Einsum: bdm, dmh -> bdh
        h = torch.einsum('bdm,dmh->bdh', pre_acts_history, self.w1) + self.b1
        h = F.relu(h)
        
        # Layer 2
        # Input: (B, D, H)
        # Weights: (D, H, 1)
        # Output: (B, D, 1)
        # Einsum: bdh, dhk -> bdk
        out = torch.einsum('bdh,dhk->bdk', h, self.w2) + self.b2
        
        return out.squeeze(-1) # [Batch, d_model]


class Synapse(nn.Module):
    """
    Synapse Model: Shares information across neurons.
    Input: Concatenation of [z^t, attention_out^t]
    Output: Pre-activations a^t
    """
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model) # Output pre-activations
        )
        
    def forward(self, x):
        return self.net(x)


class StockPredictor(nn.Module):
    """
    Continuous Thought Machine (CTM) Architecture adapted for Stock Prediction.
    """
    def __init__(self, input_shape, hp):
        super().__init__()
        self.input_shape = input_shape
        seq_len, num_features = input_shape
        
        # Hyperparameters with defaults
        self.d_model = hp.get('d_model', 128) # D
        self.memory_length = hp.get('memory_length', 10) # M
        self.n_ticks = hp.get('n_ticks', 5) # T
        self.n_heads = hp.get('n_heads', 4)
        self.nlm_hidden = hp.get('nlm_hidden', 32)
        self.dropout = hp.get('dropout', 0.1)
        self.j_out = hp.get('j_out', 32) # Number of pairs for output projection
        self.j_action = hp.get('j_action', 32) # Number of pairs for action projection
        
        # Feature Extractor (Project input sequence to Keys/Values)
        # Input: (B, Seq, Features) -> (B, Seq, d_model)
        self.feature_extractor = nn.Linear(num_features, self.d_model)
        
        # Learnable Initial Parameters
        self.z_init = nn.Parameter(torch.randn(1, self.d_model))
        self.pre_acts_history_init = nn.Parameter(torch.randn(1, self.d_model, self.memory_length))
        
        # Components
        self.synapse = Synapse(self.d_model * 2, self.d_model, self.dropout) # Input is z + attn_out
        self.nlms = NLM(self.d_model, self.memory_length, self.nlm_hidden)
        self.attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)
        
        # Projections from Synchronization to Queries/Outputs
        # Synch matrix is D_subset x D_subset? No, paper says it projects from vector S.
        # We use random pairing subsampling.
        # We select J pairs. S is size J.
        self.output_pairs_idx = None # Will init in forward or init
        self.action_pairs_idx = None
        
        self.w_mean = nn.Linear(self.j_out, 1) # Mean (Linear)
        self.w_log_sigma = nn.Linear(self.j_out, 1) # LogSigma
        self.w_action = nn.Linear(self.j_action, self.d_model) # Query vector
        
        # Synchronization Decay parameters (one per pair)
        self.r_out = nn.Parameter(torch.zeros(self.j_out))
        self.r_action = nn.Parameter(torch.zeros(self.j_action))
        
    def _init_pairs(self, device):
        if self.output_pairs_idx is None:
            # Randomly select pairs of neurons (i, j)
            self.output_pairs_idx = (
                torch.randint(0, self.d_model, (self.j_out,), device=device),
                torch.randint(0, self.d_model, (self.j_out,), device=device)
            )
            self.action_pairs_idx = (
                torch.randint(0, self.d_model, (self.j_action,), device=device),
                torch.randint(0, self.d_model, (self.j_action,), device=device)
            )

    def _compute_synch(self, z_t, z_trace, r, pairs_idx):
        """
        Recursive synchronization update.
        S^t = alpha^t / sqrt(beta^t)
        alpha^t = e^-r * alpha^{t-1} + z_i * z_j
        beta^t = e^-r * beta^{t-1} + 1
        
        z_trace stores [alpha, beta] for the specific pairs.
        Shape of z_trace: (B, J, 2)
        """
        idx_i, idx_j = pairs_idx
        
        # Gather z for selected pairs: (B, J)
        z_i = z_t[:, idx_i] 
        z_j = z_t[:, idx_j]
        
        # Decay factor: (J,) -> (1, J)
        decay = torch.exp(-torch.relu(r)).unsqueeze(0)
        
        # Update alpha (idx 0) and beta (idx 1)
        alpha_prev = z_trace[:, :, 0]
        beta_prev = z_trace[:, :, 1]
        
        alpha_new = decay * alpha_prev + (z_i * z_j)
        beta_new = decay * beta_prev + 1.0
        
        # Update trace
        z_trace_new = torch.stack([alpha_new, beta_new], dim=2)
        
        # Compute S
        # Add epsilon to beta for stability
        s = alpha_new / torch.sqrt(beta_new + 1e-6)
        
        return s, z_trace_new

    def forward(self, x, return_all_ticks=False):
        batch_size = x.size(0)
        device = x.device
        self._init_pairs(device)
        
        # Feature Extraction (Keys/Values)
        kv = self.feature_extractor(x) # (B, Seq, D)
        
        # Initialize State
        z = self.z_init.repeat(batch_size, 1) # (B, D)
        pre_acts_history = self.pre_acts_history_init.repeat(batch_size, 1, 1) # (B, D, M)
        
        # Initialize Synchronization Traces (Alpha, Beta)
        # (B, J, 2)
        trace_out = torch.zeros(batch_size, self.j_out, 2, device=device)
        trace_out[:, :, 1] = 1.0 # Beta init to 1? Paper says beta^1 = 1. Wait, recursive formula beta^{t+1} = ... + 1. If t=1, beta^1=1. So beta^0=0? 
        # Eq 17: beta^{t+1} = e^-r * beta^t + 1.
        # If we start at t=0 (before loop), beta=0. Then step 1 gives beta=1.
        trace_out[:, :, 1] = 0.0
        
        trace_action = torch.zeros(batch_size, self.j_action, 2, device=device)
        trace_action[:, :, 1] = 0.0
        
        outputs_sequence = []
        sigmas_sequence = []
        
        # Initial Action Sync (to generate first query)
        # We need an initial S_action.
        # Assume z^0 exists. Update trace with z^0?
        # The paper says: "We first collect post-activations... S^t = inner product..."
        # Ticks start at 1.
        
        for t in range(self.n_ticks):
            # 1. Compute Synchronization (using current z)
            s_action, trace_action = self._compute_synch(z, trace_action, self.r_action, self.action_pairs_idx)
            s_out, trace_out = self._compute_synch(z, trace_out, self.r_out, self.output_pairs_idx)
            
            # 2. Generate Outputs and Queries
            # Output: Mean (Linear), LogSigma
            mean = self.w_mean(s_out)
            log_sigma = self.w_log_sigma(s_out)
            sigma = torch.exp(log_sigma)
            
            outputs_sequence.append(mean)
            sigmas_sequence.append(sigma)
            
            # Query: (B, D)
            q = self.w_action(s_action).unsqueeze(1) # (B, 1, D)
            
            # 3. Attention
            # Q: (B, 1, D), K, V: (B, Seq, D)
            attn_out, _ = self.attention(q, kv, kv)
            attn_out = attn_out.squeeze(1) # (B, D)
            
            # 4. Synapse
            # Input: concat(z, attn_out)
            syn_in = torch.cat([z, attn_out], dim=1) # (B, 2D)
            a_new = self.synapse(syn_in) # (B, D)
            
            # 5. Update Pre-activation History
            # Shift history: remove oldest, add new a
            # History: (B, D, M)
            pre_acts_history = torch.cat([pre_acts_history[:, :, 1:], a_new.unsqueeze(2)], dim=2)
            
            # 6. NLMs -> New z
            z = self.nlms(pre_acts_history) # (B, D)
            
        # Stack outputs
        outputs = torch.stack(outputs_sequence, dim=1) # (B, T, 1)
        sigmas = torch.stack(sigmas_sequence, dim=1) # (B, T, 1)
        
        if return_all_ticks:
            return outputs, sigmas
        
        # For inference, return the prediction with highest certainty (min sigma)
        # sigmas: (B, T, 1)
        min_sigma_vals, min_sigma_idx = torch.min(sigmas, dim=1) # (B, 1), (B, 1)
        
        # Gather outputs at min_sigma_idx
        # outputs: (B, T, 1)
        best_out = torch.gather(outputs, 1, min_sigma_idx.unsqueeze(2)).squeeze(1) # (B, 1)
        
        return best_out
