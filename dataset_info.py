from pathlib import Path
import SimpleITK as sitk


# ============================================================
# HaN-Seg Dataset Information
# ============================================================

# Your actual HaN-Seg dataset path
DATASET_PATH = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\HaN-Seg\HaN-Seg"
)


# ============================================================
# Check dataset path
# ============================================================

print("=" * 70)
print("HaN-Seg DATASET INFORMATION")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)

if not DATASET_PATH.exists():
    print("\nERROR: Dataset path does not exist!")
    print("Please check the path.")
    exit()


# ============================================================
# Find CT files
# ============================================================

ct_files = sorted(
    DATASET_PATH.rglob("*_IMG_CT.nrrd")
)

print("\nNumber of CT volumes:", len(ct_files))


if len(ct_files) == 0:
    print("\nWARNING: No CT files were found.")
    print("\nFiles/folders found inside dataset:")

    for item in DATASET_PATH.rglob("*"):
        print(item)

    exit()


# ============================================================
# Calculate total dataset size
# ============================================================

total_size_bytes = 0

for file in DATASET_PATH.rglob("*"):

    if file.is_file():

        total_size_bytes += file.stat().st_size


total_gb = total_size_bytes / (1024 ** 3)
total_mb = total_size_bytes / (1024 ** 2)


print(f"Total dataset size: {total_gb:.2f} GB")
print(f"Total dataset size: {total_mb:.2f} MB")


# ============================================================
# CT CASE INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("CT CASE INFORMATION")
print("=" * 70)


total_slices = 0

min_slices = None
max_slices = None

min_size = None
max_size = None


for index, ct_file in enumerate(ct_files, start=1):

    print("\n")
    print(f"CASE {index}")
    print("-" * 70)

    # --------------------------------------------------------
    # File name
    # --------------------------------------------------------

    print("CT file:")
    print(ct_file.name)


    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    file_size_mb = ct_file.stat().st_size / (1024 ** 2)

    print(f"File size: {file_size_mb:.2f} MB")


    # --------------------------------------------------------
    # Read NRRD
    # --------------------------------------------------------

    try:

        image = sitk.ReadImage(str(ct_file))

    except Exception as e:

        print("ERROR reading file:")
        print(e)

        continue


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    size = image.GetSize()

    width = size[0]
    height = size[1]
    slices = size[2]


    print(
        f"Image size (X, Y, Z): "
        f"{width} × {height} × {slices}"
    )

    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Number of slices: {slices}")


    # --------------------------------------------------------
    # Voxel spacing
    # --------------------------------------------------------

    spacing = image.GetSpacing()

    print(
        f"Spacing (mm): "
        f"{spacing[0]:.4f}, "
        f"{spacing[1]:.4f}, "
        f"{spacing[2]:.4f}"
    )


    # --------------------------------------------------------
    # Physical dimensions
    # --------------------------------------------------------

    physical_width = width * spacing[0]
    physical_height = height * spacing[1]
    physical_depth = slices * spacing[2]


    print(
        "Physical size (mm): "
        f"{physical_width:.2f} × "
        f"{physical_height:.2f} × "
        f"{physical_depth:.2f}"
    )


    # --------------------------------------------------------
    # Data type
    # --------------------------------------------------------

    array = sitk.GetArrayViewFromImage(image)

    print("Pixel data type:", array.dtype)


    # --------------------------------------------------------
    # Intensity range
    # --------------------------------------------------------

    print(f"Minimum intensity: {array.min()}")
    print(f"Maximum intensity: {array.max()}")


    # --------------------------------------------------------
    # Total slices
    # --------------------------------------------------------

    total_slices += slices


    # --------------------------------------------------------
    # Minimum / Maximum slices
    # --------------------------------------------------------

    if min_slices is None or slices < min_slices:
        min_slices = slices

    if max_slices is None or slices > max_slices:
        max_slices = slices


    # --------------------------------------------------------
    # Minimum / Maximum dimensions
    # --------------------------------------------------------

    current_pixels = width * height

    if min_size is None:
        min_size = current_pixels

    if max_size is None:
        max_size = current_pixels

    if current_pixels < min_size:
        min_size = current_pixels

    if current_pixels > max_size:
        max_size = current_pixels


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal CT volumes/cases : {len(ct_files)}")

print(f"Total CT slices        : {total_slices}")

if len(ct_files) > 0:

    average_slices = total_slices / len(ct_files)

    print(
        f"Average slices/case   : "
        f"{average_slices:.2f}"
    )

    print(
        f"Minimum slices/case   : "
        f"{min_slices}"
    )

    print(
        f"Maximum slices/case   : "
        f"{max_slices}"
    )


print(f"\nTotal dataset size     : {total_gb:.2f} GB")

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)