from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import random


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\split_dataset"
)

CACHE_ROOT = Path(
    r"C:\Users\Sravani\Desktop\HaN_Seg_Project\fast_cache"
)

IMAGE_SIZE = 256

# Number of slices before and after Brainstem slices
NEIGHBOR_SLICES = 5

# Background : selected positive/nearby slices
BACKGROUND_RATIO = 1.0

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(case_dir, output_dir, select_training_slices):

    ct_files = list(
        case_dir.glob("*_IMG_CT.nrrd")
    )

    mask_files = list(
        case_dir.glob("*_OAR_Brainstem.seg.nrrd")
    )

    if not ct_files or not mask_files:
        print("Skipping:", case_dir.name)
        return 0

    ct_path = ct_files[0]
    mask_path = mask_files[0]

    # --------------------------------------------------------
    # READ VOLUME ONLY ONCE
    # --------------------------------------------------------

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
    ).astype(np.float32)

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    mean = ct.mean()
    std = ct.std()

    if std > 0:
        ct = (ct - mean) / std
    else:
        ct = ct - mean

    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    mask = (
        mask > 0
    ).astype(np.float32)

    total_slices = ct.shape[0]

    # --------------------------------------------------------
    # FIND POSITIVE BRAINSTEM SLICES
    # --------------------------------------------------------

    positive_slices = []

    for z in range(total_slices):

        if np.any(mask[z] > 0):
            positive_slices.append(z)

    # --------------------------------------------------------
    # SELECT RELEVANT SLICES
    # --------------------------------------------------------

    selected_slices = set()

    # Positive slices
    for z in positive_slices:

        selected_slices.add(z)

        # Neighboring slices
        for offset in range(
            -NEIGHBOR_SLICES,
            NEIGHBOR_SLICES + 1
        ):

            neighbor = z + offset

            if (
                0 <= neighbor < total_slices
            ):
                selected_slices.add(
                    neighbor
                )

    selected_slices = sorted(
        selected_slices
    )

    # --------------------------------------------------------
    # BACKGROUND SLICES
    # --------------------------------------------------------

    positive_count = len(
        selected_slices
    )

    background_candidates = [
        z
        for z in range(total_slices)
        if z not in selected_slices
    ]

    max_background = int(
        positive_count
        * BACKGROUND_RATIO
    )

    if len(background_candidates) > max_background:

        background_slices = random.sample(
            background_candidates,
            max_background
        )

    else:

        background_slices = (
            background_candidates
        )

    selected_slices.extend(
        background_slices
    )

    selected_slices = sorted(
        set(selected_slices)
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    case_output = (
        output_dir / case_dir.name
    )

    case_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE SELECTED SLICES
    # --------------------------------------------------------

    saved = 0

    for z in selected_slices:

        image_slice = torch.from_numpy(
            ct[z]
        ).float().unsqueeze(0)

        mask_slice = torch.from_numpy(
            mask[z]
        ).float().unsqueeze(0)

        # CT resize
        image_slice = F.interpolate(
            image_slice.unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Mask resize
        mask_slice = F.interpolate(
            mask_slice.unsqueeze(0),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="nearest"
        ).squeeze(0)

        # Save image
        np.save(
            case_output /
            f"image_{z:04d}.npy",

            image_slice.squeeze(0)
            .numpy()
            .astype(np.float16)
        )

        # Save mask
        np.save(
            case_output /
            f"mask_{z:04d}.npy",

            mask_slice.squeeze(0)
            .numpy()
            .astype(np.uint8)
        )

        saved += 1

    print(
        f"{case_dir.name}: "
        f"total={total_slices}, "
        f"positive={len(positive_slices)}, "
        f"selected={saved}"
    )

    return saved


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(split):

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
    print(
        f"PROCESSING {split.upper()}"
    )
    print("=" * 70)

    total_saved = 0

    for case_dir in case_dirs:

        # Training gets selected slices
        if split == "train":

            saved = process_case(
                case_dir,
                output_dir,
                True
            )

        # Validation/Test keep all slices
        else:

            saved = process_case_all(
                case_dir,
                output_dir
            )

        total_saved += saved

    print("\n" + "=" * 70)
    print(
        f"{split.upper()} COMPLETED"
    )
    print(
        "Total saved slices:",
        total_saved
    )
    print("=" * 70)


# ============================================================
# ALL SLICES FOR VALIDATION / TEST
# ============================================================

def process_case_all(case_dir, output_dir):

    ct_files = list(
        case_dir.glob("*_IMG_CT.nrrd")
    )

    mask_files = list(
        case_dir.glob("*_OAR_Brainstem.seg.nrrd")
    )

    if not ct_files or not mask_files:
        print("Skipping:", case_dir.name)
        return 0

    ct_image = sitk.ReadImage(
        str(ct_files[0])
    )

    mask_image = sitk.ReadImage(
        str(mask_files[0])
    )

    ct = sitk.GetArrayFromImage(
        ct_image
    ).astype(np.float32)

    mask = sitk.GetArrayFromImage(
        mask_image
    ).astype(np.float32)

    # Normalize
    mean = ct.mean()
    std = ct.std()

    if std > 0:
        ct = (ct - mean) / std
    else:
        ct = ct - mean

    # Binary mask
    mask = (
        mask > 0
    ).astype(np.float32)

    case_output = (
        output_dir / case_dir.name
    )

    case_output.mkdir(
        parents=True,
        exist_ok=True
    )

    for z in range(ct.shape[0]):

        image_slice = torch.from_numpy(
            ct[z]
        ).float().unsqueeze(0)

        mask_slice = torch.from_numpy(
            mask[z]
        ).float().unsqueeze(0)

        image_slice = F.interpolate(
            image_slice.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask_slice = F.interpolate(
            mask_slice.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        ).squeeze(0)

        np.save(
            case_output /
            f"image_{z:04d}.npy",

            image_slice.squeeze(0)
            .numpy()
            .astype(np.float16)
        )

        np.save(
            case_output /
            f"mask_{z:04d}.npy",

            mask_slice.squeeze(0)
            .numpy()
            .astype(np.uint8)
        )

    print(
        f"{case_dir.name}: "
        f"{ct.shape[0]} slices"
    )

    return ct.shape[0]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_split("train")

    process_split("val")

    process_split("test")

    print("\n")
    print("=" * 70)
    print("FAST CACHE PREPARATION COMPLETED")
    print("=" * 70)