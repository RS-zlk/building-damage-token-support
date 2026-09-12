"""CNN building encoder."""

import torch.nn as nn


class ConvNormAct(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class BuildingInstanceBackbone(nn.Module):

    def __init__(self, in_channels: int = 64, feature_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(in_channels, 64),
            ConvNormAct(64, 64),
            ConvNormAct(64, 128, stride=2),
            ConvNormAct(128, 128),
            ConvNormAct(128, feature_dim, stride=2),
            ConvNormAct(feature_dim, feature_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.project_pooled(self.pool_spatial(self.forward_spatial(x)))

    def forward_spatial(self, x):
        return self.features(x)

    def pool_spatial(self, features):
        return self.pool(features).flatten(1)

    def project_pooled(self, features):
        return features
