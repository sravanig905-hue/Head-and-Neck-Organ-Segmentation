import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CT_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"

MASK_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Brainstem.seg.nrrd"


# ============================================================
# LOAD CT AND MASK
# ============================================================

ct_image = sitk.ReadImage(CT_PATH)
mask_image = sitk.ReadImage(MASK_PATH)

ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)
mask_array = sitk.GetArrayFromImage(mask_image).astype(np.uint8)

print("Original CT shape   :", ct_array.shape)
print("Original Mask shape :", mask_array.shape)


# ============================================================
# CT INTENSITY NORMALIZATION
# ============================================================

mean = ct_array.mean()
std = ct_array.std()

if std > 0:
    ct_normalized = (ct_array - mean) / std
else:
    ct_normalized = ct_array

print("\nAfter normalization:")
print("Mean :", ct_normalized.mean())
print("Std  :", ct_normalized.std())


# ============================================================
# CONVERT MASK TO BINARY
# ============================================================

mask_binary = (mask_array > 0).astype(np.uint8)

print("\nMask unique values:")
print(np.unique(mask_binary))


# ============================================================
# FIND BRAINSTEM SLICES
# ============================================================

brainstem_slices = np.where(
    mask_binary.sum(axis=(1, 2)) > 0
)[0]

print("\nBrainstem slices:")
print("First slice :", brainstem_slices[0])
print("Last slice  :", brainstem_slices[-1])
print("Total slices:", len(brainstem_slices))


# ============================================================
# SELECT MIDDLE BRAINSTEM SLICE
# ============================================================

slice_index = brainstem_slices[len(brainstem_slices) // 2]

ct_slice = ct_normalized[slice_index]
mask_slice = mask_binary[slice_index]


# ============================================================
# VISUALIZE PREPROCESSED DATA
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(ct_slice, cmap="gray")

plt.title("Normalized CT")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(mask_slice, cmap="gray")

plt.title("Binary Brainstem Mask")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(ct_slice, cmap="gray")

plt.imshow(
    np.ma.masked_where(
        mask_slice == 0,
        mask_slice
    ),
    cmap="autumn",
    alpha=0.5
)

plt.title("Preprocessed Overlay")

plt.axis("off")


plt.tight_layout()

plt.show()