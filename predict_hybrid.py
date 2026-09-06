import torch
import numpy as np
import matplotlib.pyplot as plt

from model.hybrid_unet_transformer import HybridUNetTransformer
from slice_dataset import SliceDataset


# =========================
# SETTINGS
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "hybrid_unet_transformer.pth"
TEST_PATH = "fast_cache/test"

OUTPUT_PATH = "brainstem_prediction.png"


# =========================
# LOAD DATA
# =========================

print("=" * 60)
print("HYBRID U-NET + TRANSFORMER TEST PREDICTION")
print("=" * 60)

print("Device:", DEVICE)

dataset = SliceDataset(TEST_PATH)

print("Test samples:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError(
        "Test dataset is empty. Check fast_cache/test."
    )


# =========================
# LOAD MODEL
# =========================

model = HybridUNetTransformer(num_classes=1)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# =========================
# SELECT TEST IMAGE
# =========================

image, mask = dataset[0]

# image shape should be [1, H, W]
# Add batch dimension
input_image = image.unsqueeze(0).to(DEVICE)


# =========================
# PREDICTION
# =========================

with torch.no_grad():

    output = model(input_image)

    probability = torch.sigmoid(output)

    prediction = (
        probability > 0.5
    ).float()


# Remove dimensions
image_np = image.squeeze().numpy()
mask_np = mask.squeeze().numpy()
prediction_np = prediction.squeeze().cpu().numpy()


# =========================
# SAVE VISUALIZATION
# =========================

plt.figure(figsize=(15, 5))


# Original CT
plt.subplot(1, 3, 1)

plt.imshow(
    image_np,
    cmap="gray"
)

plt.title("CT Image")
plt.axis("off")


# Ground Truth
plt.subplot(1, 3, 2)

plt.imshow(
    image_np,
    cmap="gray"
)

plt.imshow(
    mask_np,
    alpha=0.5
)

plt.title("Ground Truth")
plt.axis("off")


# Prediction
plt.subplot(1, 3, 3)

plt.imshow(
    image_np,
    cmap="gray"
)

plt.imshow(
    prediction_np,
    alpha=0.5
)

plt.title("Model Prediction")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print()
print("=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

print("Prediction saved as:")
print(OUTPUT_PATH)

print("=" * 60)