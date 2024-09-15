import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, observation_dimensions: int, number_of_actions: int) -> None:
        super(DQN, self).__init__()
        # TODO implement a fully connected feed forward neural network using pytorch with:
        # observation_dimensions input neurons
        # two hidden layers of 128 neurons
        # number_of_actions output neurons
        # And Rectified Linear Units (ReLU) activations in between (only in between, not after the last layer)
        # There are different ways of doing this

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO write the according forward function
        return
