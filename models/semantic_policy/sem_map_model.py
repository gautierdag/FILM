import torch
import torch.nn as nn

from utils.model import Flatten


class UNetMulti(nn.Module):
    def __init__(
        self,
        input_shape,
        recurrent=False,
        hidden_size=512,
        downscaling=1,
        num_sem_categories=16,
    ):  # input shape is (240, 240)

        super(UNetMulti, self).__init__()

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 4, 32, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 73, 3, stride=1, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs):
        x = self.main(inputs)
        return x
