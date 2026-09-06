import torch
import numpy as np
import matplotlib.pyplot as plt

from model.hybrid_unet_transformer import HybridUNetTransformer
from slice_dataset import SliceDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "hybrid_unet_transformer.pth"
TEST_PATH = "fast_cache/test"

OUTPUT_PATH = "final_segmentation_result.png"


# Load dataset
dataset = SliceDataset(TEST_PATH)

if len(dataset) == 0:
    raise RuntimeError("Test dataset is empty.")

# Load model
model = HybridUNetTransformer(num_classes=1)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()


# Select test sample
image, mask = dataset[0]

input_image = image.unsqueeze(0).to(DEVICE)


# Prediction
with torch.no_grad():
    output = model(input_image)
    probability = torch.sigmoid(output)
    prediction = (probability > 0.5).float()


# Convert to numpy
image_np = image.squeeze().numpy()
mask_np = mask.squeeze().numpy()
prediction_np = prediction.squeeze().cpu().numpy()


# Create final visualization
plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(image_np, cmap="gray")
plt.title("Input CT Scan")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(image_np, cmap="gray")
plt.imshow(mask_np, alpha=0.5)
plt.title("Ground Truth")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(image_np, cmap="gray")
plt.imshow(prediction_np, alpha=0.5)
plt.title("Hybrid U-Net + Transformer")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print("=" * 60)
print("FINAL VISUALIZATION COMPLETED")
print("=" * 60)
print("Saved:", OUTPUT_PATH)
