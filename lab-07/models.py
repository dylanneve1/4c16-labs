import torch                                      # The main PyTorch library
import torch.nn as nn                             # Contains pytorch network building blocks (e.g., layers)
import torch.nn.functional as F                   # Contains functions for neural network operations (e.g., activation functions)

class SuperResolutionModel(nn.Module):
    def __init__(self, num_classes=10):
      # Define the layers you want to use here
      super().__init__()
      self.conv = nn.Conv2d(64,64,3,padding=1)

    def forward(self, x):
      # Define the forward pass of your network here
      output = []  # empty array, this should output your superres image
      
      return output