"""R2-BDA model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BuildingInstanceBackbone
from .decoders import MultiTaskDecisionHead
from .input_adapter import MultiSensorAdapter
from .swin_backbone import SwinTinyBuildingBackbone

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class R2BDANet(nn.Module):

    def __init__(
        self,
        sensor_configs: dict,
        adapter_dim: int = 64,
        feature_dim: int = 256,
        backbone_name: str = "conv",
        pretrained_backbone: bool = False,
        use_mask: bool = False,
        mask_mode: str | None = None,
        ring_width: int = 16,
    ):
        super().__init__()
        if mask_mode is None:
            mask_mode = "input_concat" if use_mask else "none"
        valid_mask_modes = {
            "none", "input_concat", "masked_rgb", "late_fusion", "dual_region",
            "matched_support_14x14", "global_matched_14x14",
        }
        if mask_mode not in valid_mask_modes:
            raise ValueError(
                f"Unsupported mask_mode={mask_mode}; choose from {sorted(valid_mask_modes)}."
            )
        if use_mask and mask_mode != "input_concat":
            raise ValueError("use_mask=True is only compatible with input_concat; set mask_mode instead.")
        if ring_width < 1:
            raise ValueError("ring_width must be a positive integer.")
        self.mask_mode = mask_mode
        self.use_mask = mask_mode not in {"none", "global_matched_14x14"}
        self.ring_width = ring_width
        adapter_configs = {
            sensor: channels + int(mask_mode == "input_concat")
            for sensor, channels in sensor_configs.items()
        }
        self.adapter = MultiSensorAdapter(adapter_configs, target_channels=adapter_dim)
        if backbone_name == "conv":
            if pretrained_backbone:
                raise ValueError("pretrained_backbone is only supported for swin_tiny.")
            self.backbone = BuildingInstanceBackbone(adapter_dim, feature_dim)
        elif backbone_name == "swin_tiny":
            self.backbone = SwinTinyBuildingBackbone(
                adapter_dim, feature_dim, pretrained=pretrained_backbone
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}; choose conv or swin_tiny.")
        self.backbone_name = backbone_name
        if mask_mode in {"late_fusion", "dual_region", "matched_support_14x14", "global_matched_14x14"}:
            self.mask_fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
            )
        self.head = MultiTaskDecisionHead(feature_dim)

        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, 1) # 1 logit for binary classification (source=0, target=1)
        )

    def _validate_mask(self, image, mask):
        if mask is None:
            raise ValueError("The selected model requires a mask, but forward received none.")
        if self.mask_mode == "matched_support_14x14":
            if mask.ndim != 4 or mask.shape[1] != 2 or mask.shape[2:] != (14, 14):
                raise ValueError(
                    f"matched_support_14x14 mask must be (B, 2, 14, 14); got {tuple(mask.shape)}"
                )
            if mask.shape[0] != image.shape[0]:
                raise ValueError(
                    f"Image and mask batch sizes differ: image={tuple(image.shape)}, mask={tuple(mask.shape)}"
                )
            return mask.to(device=image.device, dtype=image.dtype)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Mask must be (B, 1, H, W); got {tuple(mask.shape)}")
        if mask.shape[0] != image.shape[0] or mask.shape[2:] != image.shape[2:]:
            raise ValueError(
                f"Image and mask dimensions differ: image={tuple(image.shape)}, "
                f"mask={tuple(mask.shape)}"
            )
        return mask.to(device=image.device, dtype=image.dtype).clamp(0, 1)

    @staticmethod
    def _masked_pool(features, region):
        region = F.interpolate(region, size=features.shape[-2:], mode="nearest")
        weights = region.to(dtype=features.dtype)
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1.0)
        return (features * weights).sum(dim=(-2, -1)) / denominator

    def forward(self, image, sensor_name: str, mask=None, return_domain=False, alpha=1.0):
        if self.use_mask:
            mask = self._validate_mask(image, mask)

        if self.mask_mode == "input_concat":
            image = torch.cat((image, mask), dim=1)
            features = self.backbone(self.adapter(image, sensor_name))
        elif self.mask_mode == "masked_rgb":
            features = self.backbone(self.adapter(image * mask, sensor_name))
        elif self.mask_mode in {"late_fusion", "dual_region"}:
            spatial = self.backbone.forward_spatial(self.adapter(image, sensor_name))
            if self.mask_mode == "late_fusion":
                first = self.backbone.pool_spatial(spatial)
                second = self._masked_pool(spatial, mask)
            else:
                dilated = F.max_pool2d(
                    mask,
                    kernel_size=2 * self.ring_width + 1,
                    stride=1,
                    padding=self.ring_width,
                )
                ring = (dilated - mask).clamp(0, 1)
                first = self._masked_pool(spatial, mask)
                second = self._masked_pool(spatial, ring)
            first = self.backbone.project_pooled(first)
            second = self.backbone.project_pooled(second)
            features = self.mask_fusion(torch.cat((first, second), dim=1))
        elif self.mask_mode == "matched_support_14x14":
            spatial = self.backbone.forward_spatial_stage3(self.adapter(image, sensor_name))
            int_mask = mask[:, 0:1, :, :]
            ctx_mask = mask[:, 1:2, :, :]
            first = (spatial * int_mask).sum(dim=(-2, -1)) / int_mask.sum(dim=(-2, -1)).clamp_min(1.0)
            second = (spatial * ctx_mask).sum(dim=(-2, -1)) / ctx_mask.sum(dim=(-2, -1)).clamp_min(1.0)
            first = self.backbone.project_pooled_stage3(first)
            second = self.backbone.project_pooled_stage3(second)
            features = self.mask_fusion(torch.cat((first, second), dim=1))
        elif self.mask_mode == "global_matched_14x14":
            spatial = self.backbone.forward_spatial_stage3(self.adapter(image, sensor_name))
            pooled = spatial.mean(dim=(-2, -1))
            first = self.backbone.project_pooled_stage3(pooled)
            second = self.backbone.project_pooled_stage3(pooled)
            features = self.mask_fusion(torch.cat((first, second), dim=1))
        else:
            features = self.backbone(self.adapter(image, sensor_name))

        outputs = self.head(features)
        outputs["features"] = features

        if return_domain:
            reversed_features = GradientReversalLayer.apply(features, alpha)
            outputs["domain_logits"] = self.domain_discriminator(reversed_features).squeeze(-1)

        return outputs
