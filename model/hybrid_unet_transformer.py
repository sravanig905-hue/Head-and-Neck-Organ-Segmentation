import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        depth=2,
        dropout=0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):

        # x = [B, C, H, W]

        b, c, h, w = x.shape

        # [B, C, H, W]
        # -> [B, H*W, C]

        x = x.flatten(2).transpose(1, 2)

        x = self.transformer(x)

        # [B, H*W, C]
        # -> [B, C, H, W]

        x = x.transpose(1, 2).reshape(
            b,
            c,
            h,
            w
        )

        return x


class HybridUNetTransformer(nn.Module):

    def __init__(self, num_classes=31):

        super().__init__()

        # =====================================================
        # ENCODER
        # =====================================================

        self.enc1 = DoubleConv(
            1,
            64
        )

        self.pool1 = nn.MaxPool2d(
            2
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(
            2
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(
            2
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool4 = nn.MaxPool2d(
            2
        )

        # =====================================================
        # BOTTLENECK
        # =====================================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # Convert 1024 → 512 for Transformer

        self.to_transformer = nn.Conv2d(
            1024,
            512,
            kernel_size=1
        )

        # =====================================================
        # TRANSFORMER
        # =====================================================

        self.transformer = TransformerBlock(
            embed_dim=512,
            num_heads=8,
            depth=2
        )

        # Convert back 512 → 1024

        self.from_transformer = nn.Conv2d(
            512,
            1024,
            kernel_size=1
        )

        # =====================================================
        # DECODER
        # =====================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # =====================================================
        # MULTI-CLASS OUTPUT
        # =====================================================

        self.output = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # =====================================================
        # ENCODER
        # =====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        # =====================================================
        # BOTTLENECK
        # =====================================================

        b = self.bottleneck(
            self.pool4(e4)
        )

        # =====================================================
        # TRANSFORMER
        # =====================================================

        t = self.to_transformer(b)

        t = self.transformer(t)

        t = self.from_transformer(t)

        # Residual connection

        b = b + t

        # =====================================================
        # DECODER
        # =====================================================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # =====================================================
        # OUTPUT
        # =====================================================

        output = self.output(d1)

        return output