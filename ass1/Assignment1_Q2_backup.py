import torch
import torch.nn as nn

# Do not change the seed, as we need to compare your results with the reference output
torch.manual_seed(3120)

class MLP(nn.Module):
    """
    A simple two-layer feed-forward MLP
    
    TODO:
      - Implement the forward pass in forward().
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        self.fc2_bias = nn.Parameter(torch.zeros(output_dim))
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # TODO: Implement the forward pass using self.fc1_weight, self.fc1_bias,
        # self.fc2_weight, self.fc2_bias, and self.relu
        # Hint: Use torch.mm for matrix multiplication
        pass

def my_cross_entropy_loss(outputs, labels):
    """
    Compute the cross-entropy loss for the given outputs and labels.
    
    Args:
        outputs: [batch_size, num_classes] - Raw logits from the model
        labels: [batch_size] - Ground truth class indices
    
    TODO: Implement the cross-entropy loss manually following these steps:
      1. Apply softmax to outputs to get probabilities
      2. Get the predicted probability for the correct class
      3. Apply negative log likelihood
      4. Average over the batch
    """
    # TODO: Implement the cross entropy loss
    pass

def main():
    input_dim = 10
    hidden_dim = 20
    output_dim = 3
    batch_size = 4
    
    X = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    
    # Initialize model
    model = MLP(input_dim, hidden_dim, output_dim)
    
    # Forward pass
    outputs = model(X)
    
    # Calculate losses
    loss = my_cross_entropy_loss(outputs, y)
    
    print("Labels:", y)
    print("Outputs:", outputs)
    print("Loss:", loss)

if __name__ == "__main__":
    main()
