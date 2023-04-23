import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Input = 1, 40, 40, 40, Output = 8, 20, 20, 20
        self.conv_layer1 = nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # Input = 8, 20, 20, 20, Output = 16, 20, 20, 20
        self.conv_layer2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        # Input =  16, 20, 20, 20, Output = 16, 10, 10, 10
        self.max_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        # Input = 16, 10, 10, 10, Output = 32, 10, 10, 10
        self.conv_layer3 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        # Input = 32, 10, 10, 10, Output = 64, 10, 10, 10
        self.conv_layer4 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        # Input = 64, 10, 10, 10, Output = 64, 5, 5, 5 = 8000
        self.max_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8000, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.conv_layer2(out)
        out = self.relu1(out)
        out = self.max_pool1(out)
        out = self.conv_layer3(out)
        out = self.relu1(out)
        out = self.conv_layer4(out)
        out = self.relu1(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        out = nn.Dropout(0.5)(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
