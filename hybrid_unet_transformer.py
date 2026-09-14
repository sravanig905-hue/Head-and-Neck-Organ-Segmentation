import torch
import torch.nn as nn

from model.transformer import TransformerEncoder


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# HYBRID U-NET + TRANSFORMER
# ============================================================

class HybridUNetTransformer(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        # ---------------- ENCODER ----------------

        self.enc1 = DoubleConv(1, 64)

        self.enc2 = DoubleConv(64, 128)

        self.enc3 = DoubleConv(128, 256)

        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # ---------------- BOTTLENECK ----------------

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # ---------------- TRANSFORMER ----------------

        self.to_transformer = nn.Conv2d(
            1024,
            512,
            kernel_size=1
        )

        self.transformer = TransformerEncoder(
            embed_dim=512,
            num_heads=8,
            depth=2
        )

        self.from_transformer = nn.Conv2d(
            512,
            1024,
            kernel_size=1
        )

        # ---------------- DECODER ----------------

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

        # ---------------- OUTPUT ----------------

        self.output = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(p4)

        # ====================================================
        # TRANSFORMER
        # ====================================================

        t = self.to_transformer(b)

        B, C, H, W = t.shape

        # [B,C,H,W] -> [B,H*W,C]

        tokens = t.flatten(2).transpose(1, 2)

        tokens = self.transformer(tokens)

        # [B,H*W,C] -> [B,C,H,W]

        t = tokens.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        t = self.from_transformer(t)

        # Residual connection

        b = b + t

        # ====================================================
        # DECODER
        # ====================================================

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

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        return self.output(d1)