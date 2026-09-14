import torch

from model.hybrid_unet_transformer import (
    HybridUNetTransformer
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

model = HybridUNetTransformer(
    num_classes=1
).to(device)


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
# RESULTS
# ============================================================

print("=" * 70)

print("HYBRID U-NET + TRANSFORMER TEST")

print("=" * 70)

print(
    "Device      :",
    device
)

print(
    "Input shape :",
    x.shape
)

print(
    "Output shape:",
    output.shape
)

print(
    "Parameters  :",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print("=" * 70)

print("HYBRID MODEL TEST PASSED")