import torch

from model.hybrid_unet_transformer import HybridUNetTransformer


model = HybridUNetTransformer(
    num_classes=31
)

x = torch.randn(
    1,
    1,
    256,
    256
)

with torch.no_grad():

    y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)

print(
    "Total parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)