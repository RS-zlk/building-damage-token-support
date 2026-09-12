"""Swin-Tiny building encoder."""

import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.swin_transformer import SwinTransformer


class SwinTinyBuildingBackbone(nn.Module):

    def __init__(self, in_channels: int = 64, feature_dim: int = 256, pretrained: bool = False):
        super().__init__()
        if pretrained:
            if in_channels != 3:
                raise ValueError("Pretrained Swin-Tiny requires a three-channel adapter output.")
            self.model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.model.head = nn.Linear(self.model.head.in_features, feature_dim)
        else:
            self.model = SwinTransformer(
                patch_size=[4, 4],
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=[7, 7],
                num_classes=feature_dim,
            )
            self.model.features[0][0] = nn.Conv2d(
                in_channels,
                96,
                kernel_size=(4, 4),
                stride=(4, 4),
            )

        self.project_pooled_stage3 = nn.Linear(384, feature_dim)

    def forward(self, x):
        return self.project_pooled(self.pool_spatial(self.forward_spatial(x)))

    def forward_spatial(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        return self.model.permute(x)

    def forward_spatial_stage3(self, x):
        for i in range(6):
            x = self.model.features[i](x)
        return self.model.permute(x)

    def pool_spatial(self, features):
        return self.model.avgpool(features).flatten(1)

    def project_pooled(self, features):
        return self.model.head(features)
