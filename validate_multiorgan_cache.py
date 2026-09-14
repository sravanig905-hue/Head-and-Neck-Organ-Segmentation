import os
import glob
import numpy as np

PROJECT_ROOT = r"C:\Users\Sravani\Desktop\HaN_Seg_Project"

CACHE_PATH = os.path.join(
    PROJECT_ROOT,
    "multi_organ_cache"
)

# ============================================================
# LOAD CLASS MAPPING
# ============================================================

mapping_file = os.path.join(
    CACHE_PATH,
    "class_mapping.txt"
)

if not os.path.exists(mapping_file):
    print("ERROR: class_mapping.txt not found!")
    print("Please build the multi-organ cache first.")
    exit()

class_mapping = {}

with open(mapping_file, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        class_id, organ = line.split(" -> ", 1)

        class_mapping[int(class_id)] = organ


print("\n==============================================")
print("       MULTI-ORGAN CACHE VALIDATION")
print("==============================================")

print("\nClasses found:")
for class_id, organ in class_mapping.items():
    print(f"{class_id:2} -> {organ}")

print(
    f"\nTotal classes: {len(class_mapping)}"
)

# ============================================================
# EXPECTED CLASSES
# ============================================================

expected_classes = set(
    class_mapping.keys()
)

# ============================================================
# CHECK TRAIN / VAL / TEST
# ============================================================

for split in ["train", "val", "test"]:

    split_path = os.path.join(
        CACHE_PATH,
        split
    )

    print("\n----------------------------------------------")
    print(f"Checking: {split.upper()}")
    print("----------------------------------------------")

    if not os.path.exists(split_path):

        print("ERROR: Split folder not found!")

        continue

    case_folders = sorted(
        glob.glob(
            os.path.join(
                split_path,
                "case_*"
            )
        )
    )

    print(
        f"Cases found: {len(case_folders)}"
    )

    total_images = 0
    total_masks = 0

    all_values = set()

    for case_folder in case_folders:

        image_files = sorted(
            glob.glob(
                os.path.join(
                    case_folder,
                    "*_image.npy"
                )
            )
        )

        mask_files = sorted(
            glob.glob(
                os.path.join(
                    case_folder,
                    "*_mask.npy"
                )
            )
        )

        total_images += len(image_files)
        total_masks += len(mask_files)

        # ----------------------------------------------------
        # Check image-mask count
        # ----------------------------------------------------

        if len(image_files) != len(mask_files):

            print(
                f"WARNING: {os.path.basename(case_folder)} "
                f"image/mask mismatch!"
            )

        # ----------------------------------------------------
        # Inspect masks
        # ----------------------------------------------------

        for mask_file in mask_files:

            mask = np.load(mask_file)

            unique_values = np.unique(mask)

            all_values.update(
                unique_values.tolist()
            )

            # Check dimensions

            if mask.shape != (256, 256):

                print(
                    f"WARNING: Wrong mask shape: "
                    f"{mask_file}"
                )

    print(
        f"Total image slices: {total_images}"
    )

    print(
        f"Total mask slices : {total_masks}"
    )

    print(
        f"Unique mask values: "
        f"{sorted(all_values)}"
    )

    # --------------------------------------------------------
    # Check invalid class IDs
    # --------------------------------------------------------

    invalid_values = (
        all_values - expected_classes
    )

    if invalid_values:

        print(
            "ERROR: Invalid class IDs found:",
            sorted(invalid_values)
        )

    else:

        print(
            "Class IDs: VALID"
        )

# ============================================================
# FINISH
# ============================================================

print("\n==============================================")
print("       VALIDATION COMPLETED")
print("==============================================")

print("""
If:
  - image count == mask count
  - mask shape == (256, 256)
  - class IDs are valid

then the multi-organ cache is ready
for the next training step.
""")