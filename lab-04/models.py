import torch                                      # The main PyTorch library
import torch.nn as nn                             # Contains pytorch network building blocks (e.g., layers)
import torch.nn.functional as F                   # Contains functions for neural network operations (e.g., activation functions)

class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Minimum working example: single layer with ReLU activation
        # here we define all the layers/transformations being used.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*32*32, 32)     # Single Fully Connected linear layer
        self.fc2 = nn.Linear(32, num_classes) # Single Fully Connected linear layer


    def forward(self, x):
        # here define the forward transformation, using builtin pytorch calls
        # and also the layers defined in __init__

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Note that we are missing here the last Softmax layer.
        # In PyTorch, the last softmax layer is automatically included
        # in the nn.CrossEntropyLoss(). So do not define it here.
        # Your output tensor is simply expected to contain the logits for each
        # class.

        return x
