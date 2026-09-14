import SimpleITK as sitk
import numpy as np
from PIL import Image
import os

# Your dataset path
dataset_path = r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg\set_1"

# Case 01
case_path = os.path.join(dataset_path, "case_01")

# Find CT file
ct_file = None

for file in os.listdir(case_path):
    if file.endswith("_IMG_CT.nrrd"):
        ct_file = os.path.join(case_path, file)
        break

if ct_file is None:
    print("CT file not found!")
    exit()

print("CT file:")
print(ct_file)

# Read NRRD
image = sitk.ReadImage(ct_file)

# Convert to NumPy
volume = sitk.GetArrayFromImage(image)

print("Volume shape:", volume.shape)

# Select middle slice
slice_index = volume.shape[0] // 2

ct_slice = volume[slice_index]

# Normalize to 0-255
ct_slice = ct_slice.astype(np.float32)

min_val = ct_slice.min()
max_val = ct_slice.max()

ct_slice = (ct_slice - min_val) / (max_val - min_val)
ct_slice = (ct_slice * 255).astype(np.uint8)

# Save PNG
output_file = "ct_case01_slice.png"

Image.fromarray(ct_slice).save(output_file)

print("----------------------------------------")
print("PNG slice created successfully!")
print("File:", output_file)
print("Slice:", slice_index)
print("Size:", ct_slice.shape)
print("----------------------------------------")