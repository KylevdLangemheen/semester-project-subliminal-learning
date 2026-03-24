import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Architecture from paper: 784 -> 256 -> 256 -> 13
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 13),  # 10 Class + 3 Aux
        )

    def forward(
        self,
        # Switch to "batch ..." if you want to support both
        # [batch 1 28 28] and [batch 784]
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch 13"]:
        return self.net(self.flatten(x))


class SubliminalCNN(nn.Module):
    def __init__(self):
        super(SubliminalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 13)  # 10 classes + 3 auxiliary outputs

    def forward(
        self,
        x: Float[Tensor, "batch 1 28 28"],
    ) -> Float[Tensor, "batch 13"]:
        # TODO: Look into ways to typecheck steps inside the forward
        x: Float[Tensor, "batch 16 14 14"] = self.pool(self.relu(self.conv1(x)))
        x: Float[Tensor, "batch 32 7 7"] = self.pool(self.relu(self.conv2(x)))
        x: Float[Tensor, "batch 1568"] = x.view(x.size(0), -1)
        x: Float[Tensor, "batch 128"] = self.relu(self.fc1(x))
        x: Float[Tensor, "batch 13"] = self.fc2(x)
        return x
