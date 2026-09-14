import torch

from model.transformer import TransformerEncoder


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 1
HEIGHT = 16
WIDTH = 16
EMBED_DIM = 512


# ============================================================
# CREATE TRANSFORMER
# ============================================================

model = TransformerEncoder(
    embed_dim=EMBED_DIM,
    num_heads=8,
    depth=2
)


# ============================================================
# INPUT
# ============================================================

x = torch.randn(
    BATCH_SIZE,
    HEIGHT * WIDTH,
    EMBED_DIM
)


# ============================================================
# FORWARD PASS
# ============================================================

output = model(x)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)

print("TRANSFORMER TEST")

print("=" * 70)

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

print("TRANSFORMER TEST PASSED")