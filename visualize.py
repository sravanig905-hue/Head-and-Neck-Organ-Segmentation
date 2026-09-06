import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FILE PATHS
# ============================================================

ct_path = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"

mask_path = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Brainstem.seg.nrrd"


# ============================================================
# READ CT
# ============================================================

ct_image = sitk.ReadImage(ct_path)
ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

print("CT Shape:", ct_array.shape)


# ============================================================
# READ BRAINSTEM MASK
# ============================================================

mask_image = sitk.ReadImage(mask_path)
mask_array = sitk.GetArrayFromImage(mask_image).astype(np.float32)

print("Mask Shape:", mask_array.shape)


# ============================================================
# CHECK DIMENSIONS
# ============================================================

if ct_array.shape != mask_array.shape:
    print("ERROR: CT and Mask dimensions are different!")

else:
    print("CT and Mask dimensions match.")


# ============================================================
# FIND A SLICE WHERE BRAINSTEM EXISTS
# ============================================================

mask_slices = np.where(mask_array.sum(axis=(1, 2)) > 0)[0]

if len(mask_slices) == 0:
    print("WARNING: Brainstem mask is empty!")
    slice_index = ct_array.shape[0] // 2

else:
    # Select middle slice of the brainstem region
    slice_index = mask_slices[len(mask_slices) // 2]

print("Selected Slice:", slice_index)


# ============================================================
# GET CT AND MASK SLICE
# ============================================================

ct_slice = ct_array[slice_index]
mask_slice = mask_array[slice_index]


# ============================================================
# DISPLAY CT + MASK + OVERLAY
# ============================================================

plt.figure(figsize=(18, 6))


# ------------------------------------------------------------
# 1. CT IMAGE
# ------------------------------------------------------------

plt.subplot(1, 3, 1)

plt.imshow(ct_slice, cmap="gray")

plt.title("CT Scan")

plt.axis("off")


# ------------------------------------------------------------
# 2. BRAINSTEM MASK
# ------------------------------------------------------------

plt.subplot(1, 3, 2)

plt.imshow(mask_slice, cmap="gray")

plt.title("Brainstem Ground Truth")

plt.axis("off")


# ------------------------------------------------------------
# 3. OVERLAY
# ------------------------------------------------------------

plt.subplot(1, 3, 3)

plt.imshow(ct_slice, cmap="gray")

plt.imshow(
    np.ma.masked_where(mask_slice == 0, mask_slice),
    cmap="autumn",
    alpha=0.5
)

plt.title("CT + Brainstem Overlay")

plt.axis("off")


plt.tight_layout()

plt.show()