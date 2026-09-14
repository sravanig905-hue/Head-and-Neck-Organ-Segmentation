from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset"
)

CACHE_ROOT = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\cached_dataset"
)

TARGET_ORGAN = "Brainstem"
IMAGE_SIZE = 256


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def prepare_split(split):

    source_dir = DATASET_ROOT / split
    output_dir = CACHE_ROOT / split

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    case_dirs = sorted(
        source_dir.glob("case_*")
    )

    print("\n" + "=" * 70)
    print(f"PROCESSING {split.upper()}")
    print("=" * 70)

    total_slices = 0

    for case_number, case_dir in enumerate(case_dirs, 1):

        ct_files = list(
            case_dir.glob("*_IMG_CT.nrrd")
        )

        mask_files = list(
            case_dir.glob(
                f"*_OAR_{TARGET_ORGAN}.seg.nrrd"
            )
        )

        if not ct_files or not mask_files:
            print(
                f"Skipping {case_dir.name}"
            )
            continue

        ct_path = ct_files[0]
        mask_path = mask_files[0]

        print(
            f"\n[{case_number}/{len(case_dirs)}] "
            f"{case_dir.name}"
        )

        # ----------------------------------------------------
        # READ VOLUME ONLY ONCE
        # ----------------------------------------------------

        ct_image = sitk.ReadImage(
            str(ct_path)
        )

        mask_image = sitk.ReadImage(
            str(mask_path)
        )

        ct = sitk.GetArrayFromImage(
            ct_image
        ).astype(np.float32)

        mask = sitk.GetArrayFromImage(
            mask_image
        )

        print(
            "Original shape:",
            ct.shape
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        mean = ct.mean()
        std = ct.std()

        if std > 0:
            ct = (ct - mean) / std
        else:
            ct = ct - mean

        # ----------------------------------------------------
        # BINARY MASK
        # ----------------------------------------------------

        mask = (
            mask > 0
        ).astype(np.float32)

        # ----------------------------------------------------
        # OUTPUT FOLDER
        # ----------------------------------------------------

        case_output = (
            output_dir / case_dir.name
        )

        case_output.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # PROCESS EACH SLICE
        # ----------------------------------------------------

        for z in range(ct.shape[0]):

            image_slice = torch.from_numpy(
                ct[z]
            ).float().unsqueeze(0)

            mask_slice = torch.from_numpy(
                mask[z]
            ).float().unsqueeze(0)

            # Resize CT
            image_slice = F.interpolate(
                image_slice.unsqueeze(0),
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            # Resize mask
            mask_slice = F.interpolate(
                mask_slice.unsqueeze(0),
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                mode="nearest"
            ).squeeze(0)

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            image_array = (
                image_slice.squeeze(0)
                .numpy()
                .astype(np.float16)
            )

            mask_array = (
                mask_slice.squeeze(0)
                .numpy()
                .astype(np.uint8)
            )

            np.save(
                case_output /
                f"image_{z:04d}.npy",
                image_array
            )

            np.save(
                case_output /
                f"mask_{z:04d}.npy",
                mask_array
            )

        total_slices += ct.shape[0]

        print(
            f"Saved {ct.shape[0]} slices"
        )

    print("\n" + "=" * 70)
    print(f"{split.upper()} COMPLETED")
    print("Total slices:", total_slices)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_split("train")

    prepare_split("val")

    prepare_split("test")

    print("\n" + "=" * 70)
    print("CACHE PREPARATION COMPLETED")
    print("=" * 70)