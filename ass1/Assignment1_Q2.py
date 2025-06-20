import torch
import torch.nn as nn

# Do not change the seed, as we need to compare your results with the reference output
torch.manual_seed(3120)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        self.fc2_bias = nn.Parameter(torch.zeros(output_dim))
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = torch.mm(x, self.fc1_weight.t()) + self.fc1_bias
        h = self.relu(h)
        out = torch.mm(h, self.fc2_weight.t()) + self.fc2_bias
        return out


def my_cross_entropy_loss(outputs, labels):
    """
    Compute the cross-entropy loss for the given outputs and labels.

    Args:
        outputs: [batch_size, num_classes] - Raw logits from the model
        labels: [batch_size] - Ground truth class indices

    """

    probs = torch.softmax(outputs, dim=1)
    batch_size = outputs.size(0)

    selected_probs = probs[torch.arange(batch_size), labels]
    loss = -torch.log(selected_probs).mean()
    return loss



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