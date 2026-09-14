import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def horizontal_flip(image, mask):
    image = np.flip(image, axis=1).copy()
    mask = np.flip(mask, axis=1).copy()

    return image, mask


def vertical_flip(image, mask):
    image = np.flip(image, axis=0).copy()
    mask = np.flip(mask, axis=0).copy()

    return image, mask


def rotate_90(image, mask):
    image = np.rot90(image, k=1).copy()
    mask = np.rot90(mask, k=1).copy()

    return image, mask


# ============================================================
# LOAD ONE SAMPLE
# ============================================================

import SimpleITK as sitk

CT_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"

MASK_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Brainstem.seg.nrrd"


ct_image = sitk.ReadImage(CT_PATH)
mask_image = sitk.ReadImage(MASK_PATH)

ct = sitk.GetArrayFromImage(ct_image).astype(np.float32)
mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)

mask = (mask > 0).astype(np.uint8)


# ============================================================
# FIND BRAINSTEM SLICE
# ============================================================

valid_slices = np.where(
    mask.sum(axis=(1, 2)) > 0
)[0]

if len(valid_slices) == 0:
    raise ValueError("Brainstem mask is empty!")


slice_index = valid_slices[len(valid_slices) // 2]

image = ct[slice_index]
label = mask[slice_index]


# ============================================================
# NORMALIZE
# ============================================================

mean = image.mean()
std = image.std()

if std > 0:
    image = (image - mean) / std


# ============================================================
# APPLY AUGMENTATIONS
# ============================================================

flip_image, flip_mask = horizontal_flip(
    image,
    label
)

rotate_image, rotate_mask = rotate_90(
    image,
    label
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 10))


# Original
plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original CT")
plt.axis("off")


plt.subplot(2, 3, 2)
plt.imshow(label, cmap="gray")
plt.title("Original Mask")
plt.axis("off")


# Horizontal flip
plt.subplot(2, 3, 3)
plt.imshow(flip_image, cmap="gray")
plt.imshow(
    np.ma.masked_where(
        flip_mask == 0,
        flip_mask
    ),
    cmap="autumn",
    alpha=0.5
)
plt.title("Horizontal Flip")
plt.axis("off")


# Rotation
plt.subplot(2, 3, 4)
plt.imshow(rotate_image, cmap="gray")
plt.title("Rotated CT")
plt.axis("off")


plt.subplot(2, 3, 5)
plt.imshow(rotate_mask, cmap="gray")
plt.title("Rotated Mask")
plt.axis("off")


plt.subplot(2, 3, 6)
plt.imshow(rotate_image, cmap="gray")
plt.imshow(
    np.ma.masked_where(
        rotate_mask == 0,
        rotate_mask
    ),
    cmap="autumn",
    alpha=0.5
)
plt.title("Rotated Overlay")
plt.axis("off")


plt.tight_layout()
plt.show()