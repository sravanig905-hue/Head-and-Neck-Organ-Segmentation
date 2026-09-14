import torch

from model.unet import UNet


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# MODEL
# ============================================================

model = UNet(
    in_channels=1,
    num_classes=1
)

model = model.to(device)


# ============================================================
# TEST INPUT
# ============================================================

x = torch.randn(
    1,
    1,
    256,
    256
).to(device)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    output = model(x)


# ============================================================
# RESULT
# ============================================================

print("Input shape :", x.shape)

print("Output shape:", output.shape)

print("Model parameters:",
      sum(p.numel() for p in model.parameters()))