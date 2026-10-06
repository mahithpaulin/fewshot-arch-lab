"""Shared torch utilities: encoder, functional heads, seeding."""
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class Encoder(nn.Module):
    def __init__(self, in_dim: int, h: int = 128, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, out),
        )
        self.out_dim = out

    def forward(self, x):
        return self.net(x)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()
