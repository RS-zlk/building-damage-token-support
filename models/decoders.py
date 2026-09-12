"""Multitask prediction heads."""

import torch.nn as nn


class MultiTaskDecisionHead(nn.Module):

    def __init__(self, feature_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout))
        self.fine_head = nn.Linear(feature_dim, 4)
        self.binary_head = nn.Linear(feature_dim, 2)

    def forward(self, features):
        features = self.trunk(features)
        return {
            "logits_4class": self.fine_head(features),
            "logits_2class": self.binary_head(features),
        }
