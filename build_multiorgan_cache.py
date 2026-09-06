import os
import glob
import shutil
import numpy as np
import SimpleITK as sitk
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Sravani\Desktop\HaN_Seg_Project"

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "HaN-Seg",
    "HaN-Seg",
    "set_1"
)

SPLIT_PATH = os.path.join(
    PROJECT_ROOT,
    "splits"
)

output_case_dir = os.path.join(
    OUTPUT_DIR,
    split,
    case_name
)

os.makedirs(
    output_case_dir,
    exist_ok=True
)
# Image size used by our model
IMAGE_SIZE = (256, 256)

# ============================================================
# FIND ALL ORGANS AUTOMATICALLY
# ============================================================

print("\n==============================================")
print("   MULTI-ORGAN CACHE BUILDING")
print("==============================================")

organ_names = set()

case_folders = sorted(
    glob.glob(os.path.join(DATASET_PATH, "case_*"))
)

print(f"\nTotal cases found: {len(case_folders)}")

for case_folder in case_folders:

    mask_files = glob.glob(
        os.path.join(case_folder, "*_OAR_*.seg.nrrd")
    )

    for mask_file in mask_files:

        filename = os.path.basename(mask_file)

        organ = filename.split("_OAR_", 1)[1]
        organ = organ.replace(".seg.nrrd", "")

        organ_names.add(organ)

organ_names = sorted(organ_names)

print(f"Total organs found: {len(organ_names)}")

# ============================================================
# CLASS MAPPING
# ============================================================

# 0 = background
class_mapping = {
    organ: index
    for index, organ in enumerate(organ_names, start=1)
}

print("\nClass Mapping:")
print("----------------------------------------------")
print("0 -> Background")

for organ, class_id in class_mapping.items():
    print(f"{class_id:2} -> {organ}")

print("----------------------------------------------")
print(f"Total classes: {len(organ_names) + 1}")

# ============================================================
# SAVE CLASS MAPPING
# ============================================================

mapping_file = os.path.join(
    OUTPUT_PATH,
    "class_mapping.txt"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(mapping_file, "w", encoding="utf-8") as f:

    f.write("0 -> Background\n")

    for organ, class_id in class_mapping.items():
        f.write(f"{class_id} -> {organ}\n")

print(f"\nClass mapping saved to:")
print(mapping_file)

# ============================================================
# LOAD TRAIN / VAL / TEST SPLITS
# ============================================================

def load_split(split_name):

    split_file = os.path.join(
        SPLIT_PATH,
        f"{split_name}.txt"
    )

    if not os.path.exists(split_file):

        print(
            f"\nWARNING: {split_file} not found."
        )

        return []

    with open(split_file, "r") as f:

        cases = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return cases


train_cases = load_split("train")
val_cases = load_split("val")
test_cases = load_split("test")

print("\nDataset Split:")
print("----------------------------------------------")
print(f"Train cases : {len(train_cases)}")
print(f"Val cases   : {len(val_cases)}")
print(f"Test cases  : {len(test_cases)}")

# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(case_name, split_name):

    case_folder = os.path.join(
        DATASET_PATH,
        case_name
    )

    ct_files = glob.glob(
        os.path.join(
            case_folder,
            "*_IMG_CT.nrrd"
        )
    )

    if not ct_files:

        print(
            f"WARNING: CT not found for {case_name}"
        )

        return 0

    ct_file = ct_files[0]

    print(
        f"\nProcessing {split_name}/{case_name}"
    )

    # --------------------------------------------------------
    # Read CT
    # --------------------------------------------------------

    ct_image = sitk.ReadImage(ct_file)

    ct_array = sitk.GetArrayFromImage(
        ct_image
    ).astype(np.float32)

    # --------------------------------------------------------
    # Normalize CT
    # --------------------------------------------------------

    mean = np.mean(ct_array)
    std = np.std(ct_array)

    if std > 0:

        ct_array = (
            ct_array - mean
        ) / std

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    case_output = os.path.join(
        OUTPUT_PATH,
        split_name,
        case_name
    )

    os.makedirs(
        case_output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process every CT slice
    # --------------------------------------------------------

    total_slices = ct_array.shape[0]

    saved_slices = 0

    for slice_index in range(total_slices):

        ct_slice = ct_array[
            slice_index
        ]

        # ----------------------------------------------------
        # Create MULTI-CLASS mask
        # ----------------------------------------------------

        multi_mask = np.zeros(
            ct_slice.shape,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Load every organ mask
        # ----------------------------------------------------

        for organ, class_id in class_mapping.items():

            mask_pattern = os.path.join(
                case_folder,
                f"*_OAR_{organ}.seg.nrrd"
            )

            mask_files = glob.glob(
                mask_pattern
            )

            if not mask_files:
                continue

            mask_image = sitk.ReadImage(
                mask_files[0]
            )

            mask_array = sitk.GetArrayFromImage(
                mask_image
            )

            if slice_index >= mask_array.shape[0]:
                continue

            organ_slice = mask_array[
                slice_index
            ]

            multi_mask[
                organ_slice > 0
            ] = class_id

        # ----------------------------------------------------
        # Resize CT
        # ----------------------------------------------------

        ct_min = ct_slice.min()
        ct_max = ct_slice.max()

        if ct_max > ct_min:

            ct_uint8 = (
                (ct_slice - ct_min)
                / (ct_max - ct_min)
                * 255
            ).astype(np.uint8)

        else:

            ct_uint8 = np.zeros_like(
                ct_slice,
                dtype=np.uint8
            )

        ct_pil = Image.fromarray(
            ct_uint8
        )

        ct_pil = ct_pil.resize(
            IMAGE_SIZE,
            Image.BILINEAR
        )

        ct_resized = np.asarray(
            ct_pil,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Resize multi-class mask
        # ----------------------------------------------------

        mask_pil = Image.fromarray(
            multi_mask
        )

        mask_pil = mask_pil.resize(
            IMAGE_SIZE,
            Image.NEAREST
        )

        mask_resized = np.asarray(
            mask_pil,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        image_file = os.path.join(
            case_output,
            f"slice_{slice_index:04d}_image.npy"
        )

        mask_file = os.path.join(
            case_output,
            f"slice_{slice_index:04d}_mask.npy"
        )

        np.save(
            image_file,
            ct_resized
        )

        np.save(
            mask_file,
            mask_resized
        )

        saved_slices += 1

    print(
        f"  Saved slices: {saved_slices}"
    )

    return saved_slices


# ============================================================
# BUILD CACHE
# ============================================================

total_saved = 0

for split_name, cases in [
    ("train", train_cases),
    ("val", val_cases),
    ("test", test_cases)
]:

    print("\n==============================================")
    print(f"BUILDING {split_name.upper()} CACHE")
    print("==============================================")

    for case_name in cases:

        total_saved += process_case(
            case_name,
            split_name
        )

# ============================================================
# FINISHED
# ============================================================

print("\n==============================================")
print("   MULTI-ORGAN CACHE COMPLETED")
print("==============================================")

print(
    f"\nTotal slices saved: {total_saved}"
)

print(
    f"\nOutput directory:\n{OUTPUT_PATH}"
)

print("\nClasses:")
print(f"Background + {len(organ_names)} organs")

print("\nNext step:")
print("Modify Hybrid U-Net + Transformer")
print("to output 31 classes.")