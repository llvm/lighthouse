"""Defines a simple PyTorch model to be used in lighthouse's ingress examples."""

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def get_init_inputs():
    """Function to return args to pass to MLPModel.__init__()"""
    return ()


def get_sample_inputs():
    """Arguments to pass to MLPModel.forward()"""
    return (torch.randn(1, 10),)
