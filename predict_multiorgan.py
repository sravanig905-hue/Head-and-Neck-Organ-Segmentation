import os
import numpy as np
import torch
from PIL import Image

from model.hybrid_unet_transformer import HybridUNetTransformer
from multi_organ_dataset import MultiOrganDataset


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "hybrid_unet_transformer_multiorgan.pth"
OUTPUT_PATH = "multiorgan_prediction.png"

NUM_CLASSES = 31

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("MULTI-ORGAN PREDICTION TEST")
print("=" * 60)

dataset = MultiOrganDataset("test")

print("Test samples:", len(dataset))


# ============================================================
# LOAD MODEL
# ============================================================

print("Device:", DEVICE)
print("Loading model...")

model = HybridUNetTransformer(
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle either plain state_dict or checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# SELECT TEST SAMPLE
# ============================================================

image, mask = dataset[0]

image = image.unsqueeze(0).to(DEVICE)

print("Input shape:", image.shape)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    )

prediction = prediction.squeeze(0).cpu().numpy()

ground_truth = mask.numpy()


# ============================================================
# CHECK PREDICTED CLASSES
# ============================================================

unique_pred, counts_pred = np.unique(
    prediction,
    return_counts=True
)

print("\nPredicted classes:")

for class_id, count in zip(unique_pred, counts_pred):

    print(
        f"Class {class_id:2d} : {count} pixels"
    )


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = [
    "Background",
    "A_Carotid_L",
    "A_Carotid_R",
    "Arytenoid",
    "Bone_Mandible",
    "Brainstem",
    "BuccalMucosa",
    "Cavity_Oral",
    "Cochlea_L",
    "Cochlea_R",
    "Cricopharyngeus",
    "Esophagus_S",
    "Eye_AL",
    "Eye_AR",
    "Eye_PL",
    "Eye_PR",
    "Glnd_Lacrimal_L",
    "Glnd_Lacrimal_R",
    "Glnd_Submand_L",
    "Glnd_Submand_R",
    "Glnd_Thyroid",
    "Glottis",
    "Larynx_SG",
    "Lips",
    "OpticChiasm",
    "OpticNrv_L",
    "OpticNrv_R",
    "Parotid_L",
    "Parotid_R",
    "Pituitary",
    "SpinalCord"
]


print("\nPredicted organ classes:")

for class_id in sorted(unique_pred):

    print(
        f"{class_id:2d} -> {class_names[class_id]}"
    )


# ============================================================
# GROUND TRUTH CLASSES
# ============================================================

unique_gt = np.unique(ground_truth)

print("\nGround-truth classes in this slice:")

for class_id in unique_gt:

    print(
        f"{class_id:2d} -> {class_names[class_id]}"
    )


# ============================================================
# SAVE PREDICTION IMAGE
# ============================================================

# Convert class IDs to visible grayscale values
prediction_image = (
    prediction.astype(np.float32)
    / (NUM_CLASSES - 1)
    * 255
).astype(np.uint8)

Image.fromarray(
    prediction_image
).save(OUTPUT_PATH)

print("\nPrediction image saved:")
print(os.path.abspath(OUTPUT_PATH))


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION TEST COMPLETED")
print("=" * 60)