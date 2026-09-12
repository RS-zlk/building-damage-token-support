"""Sensor-specific input adapters."""

import torch
import torch.nn as nn

class Conv1x1Adapter(nn.Module):
    def __init__(self, in_channels, target_channels=64):
        super().__init__()
        if in_channels == 3 and target_channels == 3:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.GELU()
            )

    def forward(self, x):
        return self.adapter(x)

class SensorConditionedAdapter(nn.Module):
    def __init__(self, in_channels, num_sensors, target_channels=64):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False)
        self.sensor_embedding = nn.Embedding(num_sensors, target_channels * 2)
        self.act = nn.GELU()

    def forward(self, x, sensor_id):
        feat = self.base_conv(x)

        cond = self.sensor_embedding(sensor_id)
        gamma, beta = cond.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        feat = feat * (1 + gamma) + beta
        return self.act(feat)

class MultiSensorAdapter(nn.Module):
    def __init__(self, sensor_configs, target_channels=64):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for sensor_name, in_c in sensor_configs.items():
            self.adapters[sensor_name] = Conv1x1Adapter(in_channels=in_c, target_channels=target_channels)

    def forward(self, x, sensor_name):
        if sensor_name not in self.adapters:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        return self.adapters[sensor_name](x)

if __name__ == "__main__":
    dummy_wv_ms = torch.randn(2, 8, 256, 256)
    dummy_gf_rgb = torch.randn(2, 3, 256, 256)

    configs = {
        'turkey_wv_ms': 8,
        'turkey_wv_visual_rgb': 3
    }

    gateway = MultiSensorAdapter(configs, target_channels=64)

    out_ms = gateway(dummy_wv_ms, 'turkey_wv_ms')
    out_rgb = gateway(dummy_gf_rgb, 'turkey_wv_visual_rgb')

    print(f"Output WV MS shape: {out_ms.shape}")
    print(f"Output GF RGB shape: {out_rgb.shape}")
    print("MultiSensor Adapter tests passed!")
