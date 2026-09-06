import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CT_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"

MASK_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Brainstem.seg.nrrd"


# ============================================================
# TARGET SPACING
# ============================================================

TARGET_SPACING = (1.0, 1.0, 2.0)


# ============================================================
# RESAMPLING FUNCTION
# ============================================================

def resample_image(image, target_spacing, is_mask=False):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(
            original_size[i] *
            original_spacing[i] /
            target_spacing[i]
        ))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    # CT → Linear interpolation
    # Mask → Nearest Neighbor interpolation

    if is_mask:
        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor
        )
    else:
        resampler.SetInterpolator(
            sitk.sitkLinear
        )

    return resampler.Execute(image)


# ============================================================
# READ CT AND MASK
# ============================================================

ct_image = sitk.ReadImage(CT_PATH)

mask_image = sitk.ReadImage(MASK_PATH)


print("Original CT:")
print("Size:", ct_image.GetSize())
print("Spacing:", ct_image.GetSpacing())


print("\nOriginal Mask:")
print("Size:", mask_image.GetSize())
print("Spacing:", mask_image.GetSpacing())


# ============================================================
# RESAMPLE
# ============================================================

ct_resampled = resample_image(
    ct_image,
    TARGET_SPACING,
    is_mask=False
)

mask_resampled = resample_image(
    mask_image,
    TARGET_SPACING,
    is_mask=True
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

ct_array = sitk.GetArrayFromImage(
    ct_resampled
).astype(np.float32)

mask_array = sitk.GetArrayFromImage(
    mask_resampled
).astype(np.uint8)


# ============================================================
# BINARY MASK
# ============================================================

mask_array = (mask_array > 0).astype(np.uint8)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nAfter Resampling:")

print("CT size:")
print(ct_resampled.GetSize())

print("CT spacing:")
print(ct_resampled.GetSpacing())

print("\nMask size:")
print(mask_resampled.GetSize())

print("Mask spacing:")
print(mask_resampled.GetSpacing())


# ============================================================
# FIND BRAINSTEM SLICES
# ============================================================

brainstem_slices = np.where(
    mask_array.sum(axis=(1, 2)) > 0
)[0]


if len(brainstem_slices) == 0:

    print("\nERROR: Brainstem mask is empty!")

    exit()


slice_index = brainstem_slices[
    len(brainstem_slices) // 2
]


ct_slice = ct_array[slice_index]

mask_slice = mask_array[slice_index]


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.title("Resampled CT")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    mask_slice,
    cmap="gray"
)

plt.title("Resampled Brainstem Mask")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    ct_slice,
    cmap="gray"
)

plt.imshow(
    np.ma.masked_where(
        mask_slice == 0,
        mask_slice
    ),
    cmap="autumn",
    alpha=0.5
)

plt.title("Resampled CT + Mask")

plt.axis("off")


plt.tight_layout()

plt.show()