import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


CT_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"

MASK_PATH = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Brainstem.seg.nrrd"


# ------------------------------------------------------------
# Target size
# ------------------------------------------------------------

TARGET_SIZE = (256, 256)


# ------------------------------------------------------------
# Read images
# ------------------------------------------------------------

ct_image = sitk.ReadImage(CT_PATH)
mask_image = sitk.ReadImage(MASK_PATH)


# ------------------------------------------------------------
# Resample to common spacing
# ------------------------------------------------------------

def resample(image, spacing, is_mask=False):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(
            original_size[i] *
            original_spacing[i] /
            spacing[i]
        ))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor
        )
    else:
        resampler.SetInterpolator(
            sitk.sitkLinear
        )

    return resampler.Execute(image)


TARGET_SPACING = (1.0, 1.0, 2.0)

ct_resampled = resample(
    ct_image,
    TARGET_SPACING,
    False
)

mask_resampled = resample(
    mask_image,
    TARGET_SPACING,
    True
)


# ------------------------------------------------------------
# Convert to NumPy
# ------------------------------------------------------------

ct = sitk.GetArrayFromImage(
    ct_resampled
).astype(np.float32)

mask = sitk.GetArrayFromImage(
    mask_resampled
).astype(np.uint8)

mask = (mask > 0).astype(np.uint8)


print("Resampled CT:", ct.shape)
print("Resampled Mask:", mask.shape)


# ------------------------------------------------------------
# Normalize CT
# ------------------------------------------------------------

mean = ct.mean()
std = ct.std()

if std > 0:
    ct = (ct - mean) / std


# ------------------------------------------------------------
# Find slices containing Brainstem
# ------------------------------------------------------------

valid_slices = np.where(
    mask.sum(axis=(1, 2)) > 0
)[0]

print("Brainstem slices:", len(valid_slices))

if len(valid_slices) == 0:
    raise ValueError("Brainstem mask is empty!")


# ------------------------------------------------------------
# Select middle Brainstem slice
# ------------------------------------------------------------

slice_index = valid_slices[len(valid_slices) // 2]

ct_slice = ct[slice_index]
mask_slice = mask[slice_index]


# ------------------------------------------------------------
# Resize CT and MASK
# ------------------------------------------------------------

ct_img = sitk.GetImageFromArray(ct_slice)
mask_img = sitk.GetImageFromArray(mask_slice)


ct_img = sitk.Resample(
    ct_img,
    [TARGET_SIZE[0], TARGET_SIZE[1]],
    sitk.Transform(),
    sitk.sitkLinear,
    ct_img.GetOrigin(),
    (
        ct_img.GetSize()[0] / TARGET_SIZE[0],
        ct_img.GetSize()[1] / TARGET_SIZE[1]
    ),
    ct_img.GetDirection(),
    0.0,
    sitk.sitkFloat32
)


mask_img = sitk.Resample(
    mask_img,
    [TARGET_SIZE[0], TARGET_SIZE[1]],
    sitk.Transform(),
    sitk.sitkNearestNeighbor,
    mask_img.GetOrigin(),
    (
        mask_img.GetSize()[0] / TARGET_SIZE[0],
        mask_img.GetSize()[1] / TARGET_SIZE[1]
    ),
    mask_img.GetDirection(),
    0,
    sitk.sitkUInt8
)


ct_resized = sitk.GetArrayFromImage(ct_img)
mask_resized = sitk.GetArrayFromImage(mask_img)


print("Final CT slice:", ct_resized.shape)
print("Final Mask slice:", mask_resized.shape)


# ------------------------------------------------------------
# Visualize
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(ct_resized, cmap="gray")
plt.title("256 × 256 CT")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(mask_resized, cmap="gray")
plt.title("256 × 256 Brainstem Mask")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(ct_resized, cmap="gray")
plt.imshow(
    np.ma.masked_where(
        mask_resized == 0,
        mask_resized
    ),
    cmap="autumn",
    alpha=0.5
)
plt.title("Final Preprocessed Data")
plt.axis("off")


plt.tight_layout()
plt.show()