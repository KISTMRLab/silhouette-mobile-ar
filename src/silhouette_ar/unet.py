"""Compressed U-Net for multi-class object segmentation (paper Section 3.1.2).

The paper compresses U-Net to 16 filters in the first level and a 192x192 RGB
input, trains for 5 epochs with Adam (learning rate 1e-4, divided by 10 every two
epochs) and reports about 2.4M parameters. The reference it compresses from is
quoted as 7.76M parameters; a 4-pooling U-Net with 32 base filters and 2x2
up-convolutions has exactly 7,760,130 parameters for two output classes, so the
paper's description fixes the topology but not every layer detail. This
implementation keeps the 4-pooling topology with 16 base filters and uses 3x3
transposed up-convolutions plus batch normalisation: ``PAPER_CONFIG`` has
2,160,194 parameters for background + one class (``silhouette-seg params``).
Pass ``up_kernel=2`` for the plain 2x2 variant (1,942,594).
"""
from __future__ import annotations

from pathlib import Path

PAPER_CONFIG = {"base_filters": 16, "depth": 4, "up_kernel": 3, "batch_norm": True, "input_size": 192}


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError("Install training support with: pip install -e .[train]") from exc
    return torch, nn


def build_unet(num_classes: int, base_filters: int = 16, depth: int = 4, up_kernel: int = 3, batch_norm: bool = True,
               in_channels: int = 3, **_ignored):
    """Return a ``torch.nn.Module`` producing (N, num_classes, H, W) logits.

    ``num_classes`` includes background. H and W must be divisible by 2**depth
    (192 works for depth 4).
    """
    torch, nn = _torch()

    def block(cin: int, cout: int):
        layers = []
        for a, b in ((cin, cout), (cout, cout)):
            layers.append(nn.Conv2d(a, b, 3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(b))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            channels = [base_filters * 2 ** level for level in range(depth + 1)]
            self.down = nn.ModuleList()
            previous = in_channels
            for width in channels:
                self.down.append(block(previous, width))
                previous = width
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ModuleList()
            self.merge = nn.ModuleList()
            for width in reversed(channels[:-1]):
                padding = (up_kernel - 1) // 2
                output_padding = 1 if up_kernel % 2 else 0
                self.up.append(nn.ConvTranspose2d(previous, width, up_kernel, stride=2, padding=padding, output_padding=output_padding))
                self.merge.append(block(2 * width, width))
                previous = width
            self.head = nn.Conv2d(previous, num_classes, 1)

        def forward(self, x):
            skips = []
            for index, layer in enumerate(self.down):
                x = layer(x)
                if index < len(self.down) - 1:
                    skips.append(x)
                    x = self.pool(x)
            for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
                x = merge(torch.cat([up(x), skip], dim=1))
            return self.head(x)

    return UNet()


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_checkpoint(path: str | Path, model, classes: list[str], config: dict, metrics: dict | None = None) -> None:
    torch, _ = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "compressed-unet", "state_dict": model.state_dict(), "classes": list(classes),
                "config": dict(config), "metrics": metrics or {}, "parameters": count_parameters(model)}, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    torch, _ = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture") != "compressed-unet":
        raise ValueError(f"{path} is not a compressed U-Net checkpoint from this repository")
    model = build_unet(len(payload["classes"]), **payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
