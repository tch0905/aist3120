import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_k = d_model
        
        # Initialize query, key, value projection layers
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_len, d_model)
        Returns: attention scores
        """

        # TODO: Complete the following steps:
        # 1. Calculate Q, K, V matrices
        # 2. Compute attention scores (dot-product)
        # 3. Apply softmax
        # Hint: Use torch.matmul for matric multiplication and F.softmax
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att_score = torch.matmul(q, k.transpose(-2,-1))/self.d_k**0.5
        t = torch.matmul(F.softmax(att_score, dim=-1), v)

        return t

if __name__ == "__main__":
    # Do not change the seed and initlization, as we need to compare your results with the reference output
    torch.manual_seed(42)

    # Initialize module
    d_model = 4
    self_attn = SelfAttention(d_model)

    # Generate input tensor
    x = torch.randn(2, 3, 4)

    # Forward pass
    attn_scores = self_attn(x)
    print("Scores shape:", attn_scores.shape)
    print("Scores:", attn_scores)
