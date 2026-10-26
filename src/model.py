"""
WarnaIn - Deep Learning Model Architectures
Supports:
1. SIGGRAPH17Colorizer: Symmetric CIE Lab U-Net with Skip Connections & Dilated Bottleneck
2. ECCV16Colorizer: Deep Semantic Chrominance Distribution Network in CIE Lab
3. EnsembleColorizer: High-Precision Fusion of U-Net Structure + Semantic Chrominance
4. ResNetUNetColorizer: Custom ResNet-18/34 U-Net for local COCO dataset training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==========================================
# 1. Pretrained CIE Lab U-Net (SIGGRAPH 17)
# ==========================================
class SIGGRAPH17Colorizer(nn.Module):
    """
    U-Net Architecture with Skip Connections operating in CIE Lab space.
    Input: L channel (B, 1, H, W) in [0, 1]
    Output: ab channels (B, 2, H, W) normalized in [-1, 1]
    """

    def __init__(self, pretrained: bool = True):
        super(SIGGRAPH17Colorizer, self).__init__()

        # Encoder Stage 1 (H, W)
        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        # Encoder Stage 2 (H/2, W/2)
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        # Encoder Stage 3 (H/4, W/4)
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        # Encoder Stage 4 (H/8, W/8)
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        # Dilated Bottleneck Stage 5 (dilation=2)
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        # Dilated Bottleneck Stage 6 (dilation=2)
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        # Bottleneck Stage 7
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )

        # U-Net Decoder Stage 8 + Skip from Stage 3
        self.model8up = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))
        self.model3short8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )

        # U-Net Decoder Stage 9 + Skip from Stage 2
        self.model9up = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True))
        self.model2short9 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )

        # U-Net Decoder Stage 10 + Skip from Stage 1
        self.model10up = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True))
        self.model1short10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Aux classification head (present in state_dict)
        self.model_class = nn.Sequential(nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=True))

        # Final Chrominance Prediction Head (a*, b*)
        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh(),
        )

        if pretrained:
            url = "https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
            self.load_state_dict(state_dict, strict=True)

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            L: Tensor (B, 1, H, W) in range [0, 1]
        Returns:
            ab: Tensor (B, 2, H, W) in range [-1, 1] (scaled to 128-range for utils.lab_to_rgb)
        """
        orig_h, orig_w = L.shape[2], L.shape[3]
        # Ensure dimensions are multiples of 8 for clean skip connections
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            L_padded = F.pad(L, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            L_padded = L

        # Convert L [0, 1] -> centered CIE L: (L*100 - 50) / 100 = L - 0.5
        in_l = L_padded - 0.5
        # Append 3 zero channels (no manual user hints -> automatic mode)
        zeros = torch.zeros((in_l.shape[0], 3, in_l.shape[2], in_l.shape[3]), device=in_l.device, dtype=in_l.dtype)
        x = torch.cat([in_l, zeros], dim=1)

        conv1_2 = self.model1(x)
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)

        out_reg = self.model_out(conv10_2)  # [-1, 1] corresponding to [-110, 110]
        if pad_h > 0 or pad_w > 0:
            out_reg = out_reg[:, :, :orig_h, :orig_w]

        # Scale from 110-norm to 128-norm used by lab_to_rgb
        return out_reg * (110.0 / 128.0)


# ==========================================
# 2. Pretrained Semantic Chrominance Prior (ECCV 16)
# ==========================================
class ECCV16Colorizer(nn.Module):
    """
    Classification-guided CIE Lab Colorizer (Zhang et al., ECCV 2016).
    Produces rich, vibrant semantic colors via 313-bin gamut distribution.
    """

    def __init__(self, pretrained: bool = True):
        super(ECCV16Colorizer, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

        if pretrained:
            url = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
            self.load_state_dict(state_dict, strict=True)

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = L.shape[2], L.shape[3]
        # ECCV16 expects normalized L: (L*100 - 50) / 100 = L - 0.5
        in_l = L - 0.5

        conv1_2 = self.model1(in_l)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)

        out_reg = self.model_out(self.softmax(conv8_3))
        out_reg = F.interpolate(out_reg, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return out_reg * (110.0 / 128.0)


# ==========================================
# 3. Ensemble Fusion Colorizer (U-Net + Semantic Prior)
# ==========================================
class EnsembleColorizer(nn.Module):
    """
    Combines SIGGRAPH17 U-Net (sharp boundaries & spatial consistency via skip connections)
    with ECCV16 (vivid semantic chrominance) for optimal visual quality.
    """

    def __init__(self, unet_weight: float = 0.65, eccv_weight: float = 0.35):
        super(EnsembleColorizer, self).__init__()
        self.unet = SIGGRAPH17Colorizer(pretrained=True)
        self.eccv = ECCV16Colorizer(pretrained=True)
        self.unet_weight = unet_weight
        self.eccv_weight = eccv_weight

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        ab_unet = self.unet(L)
        ab_eccv = self.eccv(L)
        return (self.unet_weight * ab_unet) + (self.eccv_weight * ab_eccv)


# ==========================================
# 4. Custom ResNet-18 U-Net Colorizer (For Local COCO Training)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetEncoder(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), pretrained: bool = True):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        if pretrained:
            self._load_official_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _load_official_weights(self):
        try:
            url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
            state_dict["conv1.weight"] = state_dict["conv1.weight"].mean(dim=1, keepdim=True)
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)
            self.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[WARNING] Could not load ResNet-18 weights: {e}")


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNetColorizer(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super(ResNetUNetColorizer, self).__init__()
        layers = (2, 2, 2, 2) if backbone == "resnet18" else (3, 4, 6, 3)
        self.encoder = ResNetEncoder(layers=layers, pretrained=pretrained)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 1, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        stem_out = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(L)))
        feat1 = self.encoder.layer1(self.encoder.maxpool(stem_out))
        feat2 = self.encoder.layer2(feat1)
        feat3 = self.encoder.layer3(feat2)
        feat4 = self.encoder.layer4(feat3)

        d4 = self.dec4(feat4, feat3)
        d3 = self.dec3(d4, feat2)
        d2 = self.dec2(d3, feat1)
        d1 = self.dec1(d2, stem_out)
        d0 = self.dec0(d1, L)
        return self.head(d0)


def build_colorizer(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    pretrained: bool = True,
    engine: str = "siggraph17",
) -> nn.Module:
    """
    Factory function to build the requested colorization model.
    Engines:
      - 'siggraph17': Pretrained CIE Lab U-Net with Skip Connections (High accuracy, natural tones)
      - 'ensemble': Fusion of SIGGRAPH17 U-Net + ECCV16 Semantic Prior (Rich colors + sharp edges)
      - 'eccv16': Pretrained Semantic Chrominance Distribution Network (Vivid colors)
      - 'resnet18': Custom ResNet-18 U-Net trained on local dataset (loads checkpoint_path)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if engine == "ensemble":
        model = EnsembleColorizer()
    elif engine == "siggraph17":
        model = SIGGRAPH17Colorizer(pretrained=pretrained)
    elif engine == "eccv16":
        model = ECCV16Colorizer(pretrained=pretrained)
    elif engine == "resnet18":
        model = ResNetUNetColorizer(backbone="resnet18", pretrained=pretrained)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
    else:
        # Auto-detect checkpoint architecture if a path is passed
        if checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                if "model8up.0.weight" in state_dict:
                    model = SIGGRAPH17Colorizer(pretrained=False)
                    model.load_state_dict(state_dict, strict=True)
                elif "encoder.conv1.weight" in state_dict:
                    model = SIGGRAPH17Colorizer(pretrained=True)
                else:
                    model = SIGGRAPH17Colorizer(pretrained=True)
            except Exception:
                model = SIGGRAPH17Colorizer(pretrained=True)
        else:
            model = SIGGRAPH17Colorizer(pretrained=True)

    model.to(device)
    model.eval()
    return model
