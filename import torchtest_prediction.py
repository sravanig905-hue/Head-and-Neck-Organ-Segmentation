import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk

from model.unet import UNet


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best_unet_brainstem.pth"

CT_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset\test\case_01\case_01_IMG_CT.nrrd"

MASK_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset\test\case_01\case_01_OAR_Brainstem.seg.nrrd"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("U-NET PREDICTION TEST")
print("=" * 70)

print("Device:", DEVICE)

model = UNet(num_classes=1)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("✓ Model loaded successfully")


# ============================================================
# LOAD CT
# ============================================================

ct_image = sitk.ReadImage(CT_PATH)

mask_image = sitk.ReadImage(MASK_PATH)

ct_volume = sitk.GetArrayFromImage(ct_image).astype(
    np.float32
)

mask_volume = sitk.GetArrayFromImage(mask_image)

mask_volume = (mask_volume > 0).astype(np.float32)

print("CT shape:", ct_volume.shape)

print("Mask shape:", mask_volume.shape)


# ============================================================
# FIND A SLICE CONTAINING BRAINSTEM
# ============================================================

slice_indices = np.where(
    np.sum(mask_volume, axis=(1, 2)) > 0
)[0]

if len(slice_indices) == 0:

    raise RuntimeError(
        "No Brainstem slice found!"
    )

# Select middle positive slice
z = slice_indices[len(slice_indices) // 2]

print("Selected slice:", z)


# ============================================================
# PREPROCESS SLICE
# ============================================================

ct_slice = ct_volume[z]

mean = ct_slice.mean()

std = ct_slice.std()

if std > 0:

    ct_slice = (
        ct_slice - mean
    ) / std

else:

    ct_slice = ct_slice - mean


# Convert to tensor
image = torch.from_numpy(
    ct_slice
).float().unsqueeze(0)


# Resize
image = F.interpolate(
    image.unsqueeze(0),
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)


image = image.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image)

    probability = torch.sigmoid(output)

    prediction = (
        probability > 0.5
    ).float()


# Convert to numpy
prediction = prediction.squeeze().cpu().numpy()

probability = probability.squeeze().cpu().numpy()

ground_truth = mask_volume[z]


# ============================================================
# DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(15, 5))


# Original CT
plt.subplot(1, 3, 1)

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.title("Original CT")

plt.axis("off")


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(
    ground_truth,
    cmap="gray"
)

plt.title("Ground Truth Brainstem")

plt.axis("off")


# Prediction
plt.subplot(1, 3, 3)

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.imshow(
    prediction,
    alpha=0.5
)

plt.title("U-Net Prediction")

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# DICE SCORE
# ============================================================

intersection = (
    prediction * ground_truth
).sum()

dice = (
    2 * intersection + 1e-6
) / (
    prediction.sum()
    + ground_truth.sum()
    + 1e-6
)

print("\n" + "=" * 70)

print(
    f"Dice Score: {dice:.4f}"
)

print("=" * 70)

print("PREDICTION TEST COMPLETED")