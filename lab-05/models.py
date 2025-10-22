import torch                                      # The main PyTorch library
import torch.nn as nn                             # Contains pytorch network building blocks (e.g., layers)
import torch.nn.functional as F                   # Contains functions for neural network operations (e.g., activation functions)

class ConvNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNN, self).__init__()
        self.features = nn.Sequential(
        	# Block 1
        	nn.Conv2d(3, 64, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(64, 64, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=2, stride=2),
        	# Block 2
        	nn.Conv2d(64, 128, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(128, 128, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=2, stride=2),
        	# Block 3
        	nn.Conv2d(128, 256, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(256, 256, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=2, stride=2),
        	# Block 4
        	nn.Conv2d(256, 256, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(256, 256, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
          nn.Dropout(0.5),
        	nn.Linear(256 * 2 * 2, 1024),
        	nn.ReLU(inplace=True),
          nn.Dropout(0.5),
			    nn.Linear(1024, 256),
        	nn.ReLU(inplace=True),
          nn.Dropout(0.3),
        	nn.Linear(256, num_classes),
        )


    def forward(self, x):
        # same here, look back at lab 4.
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

